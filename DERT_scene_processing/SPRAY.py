import os
import copy
import math
import pickle
import argparse
import time
import itertools
from functools import partial
from re import X
import numpy as np
import random
from functools import cmp_to_key
import multiprocessing as mp
import Lidar_find
from tqdm import tqdm
from pathlib import Path
import theory
from typing import Dict, List, Tuple
from scipy.constants import speed_of_light as c     # in m/s

RNG = np.random.default_rng(seed=42)

AVAILABLE_TAU_Hs = [20]
LIDAR_FOLDERS = os.listdir('/mnt/8tssd/AdverseWeather/particular/')
INTEGRAL_PATH = Path(os.path.dirname(os.path.realpath(
    __file__))) / 'integral_lookup_tables' / 'original'


def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR foggification')

    parser.add_argument(
        '-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-f', '--n_features',
                        help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default=str(r'/mnt/8tssd/waymo/perception/tfrecord/waymo_processed_data_train_val_test/'))
    parser.add_argument('-i', '--info_folder', help='info folder of dataset',
                        default=str(r'/mnt/8tssd/AdverseWeather/particular/'))
    parser.add_argument('-s', '--save_folder', help='info folder of dataset',
                        default=str(r'/mnt/8tssd/SPRAY/txt'))
    parser.add_argument('-sn', '--savenpy_folder', help='info folder of dataset',
                        default=str(r'/mnt/8tssd/SPRAY/npy'))
    arguments = parser.parse_args()

    return arguments


def get_available_alphas() -> List[float]:

    alphas = []

    for file in os.listdir(INTEGRAL_PATH):

        if file.endswith(".pickle"):

            alpha = file.split('_')[-1].replace('.pickle', '')

            alphas.append(float(alpha))

    return sorted(alphas)


class ParameterSet:

    def __init__(self, **kwargs) -> None:

        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        # in m              (displacement of transmitter and receiver)
        self.D = 0.1
        # in m              (radius of the transmitter aperture)
        self.ROH_T = 0.01
        # in m              (radius of the receiver aperture)
        self.ROH_R = 0.01
        # in deg            (opening angle of the transmitter's FOV)
        self.GAMMA_T_DEG = 2
        # in deg            (opening angle of the receiver's FOV)
        self.GAMMA_R_DEG = 3.5
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)

        # assert self.GAMMA_T_DEG != self.GAMMA_R_DEG, 'would lead to a division by zero in the calculation of R_2'
        #
        # self.R_1 = (self.D - self.ROH_T - self.ROH_R) / (
        #         np.tan(self.GAMMA_T / 2) + np.tan(self.GAMMA_R / 2))  # in m (see Figure 2 and Equation (11) in [1])
        # self.R_2 = (self.D - self.ROH_R + self.ROH_T) / (
        #         np.tan(self.GAMMA_R / 2) - np.tan(self.GAMMA_T / 2))  # in m (see Figure 2 and Equation (12) in [1])

        # R_2 < 10m in most sensors systems
        # co-axial setup (where R_2 = 0) is most affected by water droplet returns

        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


def get_integral_dict(p: ParameterSet) -> Dict:

    alphas = get_available_alphas()

    alpha = min(alphas, key=lambda x: abs(x - p.alpha))
    tau_h = min(AVAILABLE_TAU_Hs, key=lambda x: abs(x - int(p.tau_h * 1e9)))

    filename = INTEGRAL_PATH / \
        f'integral_0m_to_200m_stepsize_0.1m_tau_h_{tau_h}ns_alpha_{alpha}.pickle'

    with open(filename, 'rb') as handle:
        integral_dict = pickle.load(handle)

    return integral_dict


def P_R_rain_hard(p: ParameterSet, pc: np.ndarray,alpha) -> np.ndarray:
    r_0 = np.linalg.norm(pc[:, 0:3], axis=1)
    rmax = np.max(r_0)                     # max range (m)
    print(rmax)
    Pmin = 0.9* rmax **(-2)                 # min measurable power (arb units)c
    lost = np.zeros(pc.shape[0])
    bf = np.mean(pc[:,3])
    pc[:, 3] = np.exp(-2 * p.alpha * r_0) * pc[:, 3]
    
    for i in range(len(pc)):
        ref = pc[i, 3]
        ran = r_0[i]
        
        P0  = ref/(ran**2) # power
        if P0<Pmin:
            lost[i] = 1
    pc = pc[lost==0,:]
    return pc


def P_R_rain_soft(lidar : int, rain_mask: np.ndarray, pc: np.ndarray, particle: np.ndarray, particle_d: dict, original_points: np.ndarray):

    pc = copy.deepcopy(pc)
    

    r_zeros = np.linalg.norm(original_points[:, 0:3], axis=1)
    r_particle = np.linalg.norm(particle[:, 0:3], axis=1)

    num_rain_responses = 0

    cnt = 0
    cc = 0
    for i in range(particle.shape[0]):
        R = r_particle[i]
        r_max = 0
        elo = random.uniform(0,1)
        # no target add directly
        if len(particle_d[i]) == 0 and lidar != 1:
            cc += 1
            pc = np.append(pc, np.array(
                [particle[i, 0], particle[i, 1], particle[i, 2], random.uniform(0.01,0.0001), elo , -1]).reshape(1, -1), axis=0)
            rain_mask = np.append(rain_mask, np.array([int(particle[i, 3])]), axis=0)
            continue
        cc += 1
        pc_idx = particle_d[0]
        pc[pc_idx,:3] = particle[i, :3]
        rain_mask[pc_idx] = int(particle[i, 3])
    return pc, rain_mask

AlllidarPositionInWaymo = [
    [1.43, 0, 2.184],
    [4.07, 0, 0.691],
    [3.245, 1.025, 0.981],
    [3.245, -1.025,  0.981],
    [-1.154,  0,  0.466]
]


def getDistance(x):
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]


def _map(dst_folder, dstnpy_folder, parameter_set, all_paths, infos_paths, all_files, i) -> None:
    alpha = random.uniform(0.002, 0.007)
    sub_x = 0
    sub_z = 0
    infos_path = infos_paths[i]
    savetxt = False
    lidar_save_path = os.path.join(dst_folder, all_files[i][1:4] + '.txt')
    lidarnpy_save_path = os.path.join(dstnpy_folder, all_files[i][1:4] + '.npy')

    
    if os.path.exists(lidarnpy_save_path) :
       return
    points = np.load(all_paths[i])
    #points[:, 3] = np.tanh(points[:, 3])
    where_are_inf = np.isinf(points[:,3])
    points[where_are_inf, 3] = 1e4
    infos = np.loadtxt(infos_path)
    if savetxt:
        np.savetxt(os.path.join(dst_folder, all_files[i][1:4] + '_real.txt'), points[:,:3], fmt="%.2f %.2f %.2f")
    
    if len(infos)!=0:
        infos = infos[infos[:,2] > 0.02,:]
    #delete point in selfcar
        infos = infos[~((infos[:,0]>-1.4)&(infos[:,0]<4.5)&(infos[:,1]>-1)&(infos[:,1]<1)),:]
    if len(infos) == 0 :
        print( " scattered points : ", 0)
        points = np.concatenate((points, np.ones(len(points)).reshape(-1, 1)), axis=1)
        points = points.astype(np.float16)
        np.save(lidarnpy_save_path, points)
        return
    
    original_points = points
    original_points = np.concatenate((original_points, np.arange(0, original_points.shape[0], 1).reshape(-1, 1)), axis=1)
    infos = np.concatenate((infos, np.arange(0, infos.shape[0], 1).reshape(-1, 1)), axis=1)
    #label point is scatted by which particle
    rain_mask = np.zeros(len(points), dtype=int) - 1
    #label particle is scatter or not 
    scatted_idx = np.zeros(infos.shape[0])
    for idx in range(5):

        sub_x = AlllidarPositionInWaymo[idx][0]
        sub_y = AlllidarPositionInWaymo[idx][1]
        sub_z = AlllidarPositionInWaymo[idx][2]


        points = points - [sub_x, sub_y, sub_z, 0, 0, 0]
        info = infos - [sub_x, sub_y, sub_z, 0]
        original_points = original_points - [sub_x, sub_y, sub_z, 0, 0, 0, 0]
        
        

        if idx > 0:
            r_info = np.linalg.norm(info[:,:3], axis=1)
            r_pc = np.linalg.norm(original_points[:,:3], axis=1)
            info = info[r_info <= 20]
            pc = original_points[r_pc <= 20]
        else :
            pc = original_points

        if len(info)==0:
            points = points + [sub_x, sub_y, sub_z, 0, 0, 0]
            info = info + [sub_x, sub_y, sub_z, 0]
            original_points = original_points + [sub_x, sub_y, sub_z, 0, 0, 0, 0]
            continue
        
        info, d = Lidar_find.MatchPoint(info, pc, idx)
        points, rain_mask = P_R_rain_soft(lidar = idx + 1, rain_mask = rain_mask, pc=points, particle=info, particle_d=d, original_points = original_points)
        
        
        points = points + [sub_x, sub_y, sub_z, 0, 0, 0]
        info = info + [sub_x, sub_y, sub_z, 0]
        original_points = original_points + [sub_x, sub_y, sub_z, 0, 0, 0, 0]

        

    
    rain_mask[rain_mask > -1] = 0
    rain_mask[rain_mask == -1] = 1
    #print( " scattered points : ", rain_mask[rain_mask != 1].shape[0])
    points = np.concatenate((points, rain_mask.reshape(-1, 1)), axis=1)
    #print(bf, np.mean(points[:,3]))
    points = points.astype(np.float16)
    np.save(lidarnpy_save_path, points)
    if savetxt:
        np.savetxt(lidar_save_path, points[:,[0,1,2,6]], fmt="%.2f %.2f %.2f %i")
    


if __name__ == '__main__':

    args = parse_arguments()

    print('')
    print(f'using {args.n_cpus} CPUs')

    available_alphas = get_available_alphas()

    for lidar_folder in LIDAR_FOLDERS:
        
        
        src_folder = os.path.join(args.root_folder, lidar_folder)
        info_folder = os.path.join(args.info_folder, lidar_folder)
        all_files = []

        for root, dirs, files in os.walk(src_folder):
            all_files = sorted(files)
    

        print(info_folder)
        for root, dirs, files in os.walk(info_folder):
            infos_paths = sorted(files)
        print(infos_paths)
        tmp = []
        for file in all_files:
            if file.find('.npy')!=-1:
                tmp.append(file)
        
        all_files = tmp

        all_paths = [os.path.join(src_folder, file) for file in all_files]
        infos_paths = [os.path.join(info_folder, file) for file in infos_paths]

        for available_alpha in available_alphas:
            save_folder = os.path.join(args.save_folder, lidar_folder)
            # f'{save_folder}_CVL_beta_{available_alpha:.3f}'
            dst_folder = save_folder

            savenpy_folder = os.path.join(args.savenpy_folder, lidar_folder)
            # f'{savenpy_folder}_CVL_beta_{available_alpha:.3f}'
            dstnpy_folder = savenpy_folder

            Path(dst_folder).mkdir(parents=True, exist_ok=True)
            Path(dstnpy_folder).mkdir(parents=True, exist_ok=True)

            print('')
            print(f'alpha {available_alpha}')
            print('')
            print(f'searching for point clouds in    {src_folder}')
            print(f'saving augmented point clouds to {dstnpy_folder}')
            
            parameter_set = ParameterSet(alpha=available_alpha, gamma=0.000001)
            n = min(len(all_paths), len(infos_paths))
            __map = partial(_map, dst_folder, dstnpy_folder,
                            parameter_set, all_paths, infos_paths, all_files)

            with mp.Pool(40) as pool:

                l = list(tqdm(pool.imap(__map, range(0, n)), total=n))
            break
