from cmath import sqrt
import numpy as np
import os
import binascii
import struct
import math
import bisect
from functools import cmp_to_key
inclinationTable = None

epsilon = 50 / 2650.0

def initinclinationTable(flag):
    if flag == 0:
        inclinationTable = [
    #Top
        -0.30572402,
        -0.29520813,
        -0.28440472,
        -0.273567,
        -0.26308975,
        -0.2534273,
        -0.2439661,
        -0.23481996,
        -0.22545554,
        -0.21602648,
        -0.20670462,
        -0.1978009,
        -0.18938406,
        -0.18126072,
        -0.1726162,
        -0.16460761,
        -0.15673536,
        -0.14926358,
        -0.14163844,
        -0.13426943,
        -0.12712012,
        -0.12003257,
        -0.113345,
        -0.10682256,
        -0.10021271,
        -0.09397242,
        -0.08843948,
        -0.08211779,
        -0.07702542,
        -0.07161406,
        -0.06614764,
        -0.06126762,
        -0.05665334,
        -0.05186487,
        -0.0474777,
        -0.04311179,
        -0.0391097,
        -0.03539012,
        -0.03155818,
        -0.02752789,
        -0.02453311,
        -0.02142827,
        -0.01853052,
        -0.01531176,
        -0.01262421,
        -0.00942546,
        -0.00681699,
        -0.00402034,
        -0.00111514,
        0.00189587,
        0.00496216,
        0.00799484,
        0.01079605,
        0.01351344,
        0.01663751,
        0.01930892,
        0.02243253,
        0.02530314,
        0.02825533,
        0.03097034,
        0.03393894,
        0.03666058,
        0.03982922,
        0.04311189,
        ]
    else:
        inclinationTable = [
    #side
        0.5183629,
        0.5078908,
        0.49741876,
        0.48694694,
        0.47647488,
        0.46600306,
        0.455531,
        0.44505894,
        0.43458688,
        0.42411494,
        0.413643,
        0.40317106,
        0.39269912,
        0.38222718,
        0.37175512,
        0.36128318,
        0.35081124,
        0.3403393,
        0.32986724,
        0.31939518,
        0.30892324,
        0.2984513,
        0.28797936,
        0.2775073,
        0.26703537,
        0.25656343,
        0.24609149,
        0.23561943,
        0.22514749,
        0.21467555,
        0.2042036,
        0.19373155,
        0.18325949,
        0.17278755,
        0.1623156,
        0.15184367,
        0.14137161,
        0.13089967,
        0.12042773,
        0.10995579,
        0.09948385,
        0.08901179,
        0.07853985,
        0.06806791,
        0.05759585,
        0.04712379,
        0.03665185,
        0.02617991,
        0.01570797,
        0.0052360296,
        -0.0052360296,
        -0.01570797,
        -0.02617991,
        -0.03665185,
        -0.04712379,
        -0.05759585,
        -0.06806791,
        -0.07853985,
        -0.08901179,
        -0.09948385,
        -0.10995579,
        -0.12042773,
        -0.13089967,
        -0.14137161,
        -0.15184367,
        -0.1623156,
        -0.17278755,
        -0.18325949,
        -0.19373155,
        -0.2042036,
        -0.21467555,
        -0.22514749,
        -0.23561943,
        -0.24609149,
        -0.25656343,
        -0.26703537,
        -0.2775073,
        -0.28797936,
        -0.2984513,
        -0.30892324,
        -0.31939518,
        -0.32986724,
        -0.3403393,
        -0.35081124,
        -0.36128318,
        -0.37175512,
        -0.38222718,
        -0.39269912,
        -0.40317106,
        -0.413643,
        -0.42411494,
        -0.434587,
        -0.44505894,
        -0.45553088,
        -0.46600294,
        -0.476475,
        -0.48694694,
        -0.49741888,
        -0.5078908,
        -0.51836276,
        -0.5288348,
        -0.53930676,
        -0.5497787,
        -0.56025076,
        -0.5707227,
        -0.58119464,
        -0.59166664,
        -0.6021386,
        -0.6126106,
        -0.6230826,
        -0.6335546,
        -0.6440265,
        -0.65449846,
        -0.66497046,
        -0.6754424,
        -0.6859144,
        -0.6963864,
        -0.7068584,
        -0.71733034,
        -0.72780234,
        -0.7382743,
        -0.7487462,
        -0.7592183,
        -0.7696902,
        -0.7801622,
        -0.79063416,
        -0.80110615,
        -0.8115781,
        -0.82205015,
        -0.8325221,
        -0.84299403,
        -0.85346603,
        -0.863938,
        -0.87441,
        -0.884882,
        -0.895354,
        -0.9058259,
        -0.9162979,
        -0.92676985,
        -0.9372418,
        -0.9477138,
        -0.9581858,
        -0.9686578,
        -0.97912973,
        -0.98960173,
        -1.0000737,
        -1.0105456,
        -1.0210177,
        -1.0314896,
        -1.0419617,
        -1.0524335,
        -1.0629056,
        -1.0733775,
        -1.0838494,
        -1.0943215,
        -1.1047934,
        -1.1152654,
        -1.1257374,
        -1.1362094,
        -1.1466813,
        -1.1571534,
        -1.1676253,
        -1.1780972,
        -1.1885693,
        -1.1990412,
        -1.2095132,
        -1.2199852,
        -1.2304572,
        -1.2409291,
        -1.2514011,
        -1.2618731,
        -1.2723451,
        -1.282817,
        -1.2932891,
        -1.303761,
        -1.314233,
        -1.3247049,
        -1.335177,
        -1.3456489,
        -1.3561208,
        -1.3665929,
        -1.3770648,
        -1.3875368,
        -1.3980088,
        -1.4084808,
        -1.4189527,
        -1.4294246,
        -1.4398967,
        -1.4503686,
        -1.4608406,
        -1.4713126,
        -1.4817846,
        -1.4922565,
        -1.5027286,
        -1.5132005,
        -1.5236725,
        -1.5341444,
        -1.5446165,
        -1.5550884,
        -1.5655603
    ]
    inclinationTable.sort()
    return inclinationTable
def GetDistToLidarXY(x, y):
    return sqrt(x * x + y * y).real

def GetAzimuth(x, y):
    return math.atan2(y, x)

def GetInclination(z, distToLidarXY):
    return math.atan2(z, distToLidarXY)

def FindClosedInclination(inclination):
    if np.min(np.array([abs(i - inclination)  for i in inclinationTable])) < 0.01:
        return np.argmin(np.array([abs(i - inclination)  for i in inclinationTable]))
    else :
        return -1

#binary search to find those point who azimuth distance is less than epsilon
def GetValidPoints(azimuth, InclinationSet, pc):
    
    if len(InclinationSet) == 0:
        return []
    idx = bisect.bisect_left(InclinationSet[:, 0], azimuth - epsilon)
    valid_idx = []
    for i in range(idx, InclinationSet.shape[0]):
        if abs(InclinationSet[i, 0] - azimuth) > epsilon:
            break
        valid_idx.append(int(pc[int(InclinationSet[i, 1]),-1]))
    np.random.shuffle(valid_idx)
    valid_idx = valid_idx[:min(len(valid_idx),5)]
    return valid_idx

#Grouping point into different set arrcording to the most closed inclination
def FindClosedRealPoint(infos, InclinationSets, pc):
    particle_d = {}
    for i in range(len(infos)):
        x, y, z = infos[i]
        distToLidarXY = GetDistToLidarXY(x, y)
        inclination = GetInclination(z, distToLidarXY)
        #not valid
        
        if inclination > inclinationTable[-1] + 0.01 or inclination < inclinationTable[0] - 0.01:
            particle_d[i] = []
            continue
        #print(inclination, inclinationTable[-1] + 0.01, inclinationTable[0] - 0.01)
        azimuth = GetAzimuth(x, y)
        set_idx = FindClosedInclination(inclination)
        
        valid_idx = GetValidPoints(azimuth, InclinationSets[set_idx], pc)
        particle_d[i] = valid_idx
    return particle_d
        
def GetInclinationSet(pc):
    InclinationSets = [[] for i in range(300)]
    mi = 1e9
    ma = 0
    for i in range(pc.shape[0]):
        x, y, z = pc[i][:3]
        distToLidarXY = GetDistToLidarXY(x, y)
        inclination = GetInclination(z, distToLidarXY)
        azimuth = GetAzimuth(x, y)
        idx = FindClosedInclination(inclination)
        if idx !=-1:
            InclinationSets[idx].append([azimuth, i, inclination])
    for idx in range(len(InclinationSets)):
        InclinationSets[idx].sort(key=lambda s:s[0])
        InclinationSets[idx] = np.array(InclinationSets[idx])
    return InclinationSets

def getDistance(x):
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] 

def MatchPoint(infos, pc, flag):
    global inclinationTable
    inclinationTable = initinclinationTable(flag)
    InclinationSets = GetInclinationSet(pc[:,:3])
    infos = infos.tolist()
    infos.sort(key=cmp_to_key(lambda a, b: getDistance(b) - getDistance(a)))
    infos = np.array(infos)
    return infos, FindClosedRealPoint(infos[:,:3], InclinationSets, pc)


if __name__ == '__main__':
    infos_file = r'C:\Users\ylwhxht\Desktop\output\infoposition - bake\position0100.txt'
    pc_file = r'F:\code\LiDAR_fog_sim-main\data\lidar_hdl64_strongest\085.txt'
    infos = np.loadtxt(infos_file)
    pc = np.loadtxt(pc_file)[:,:4]
    pc[:,3] = 0
    d = MatchPoint(infos, pc)
    r_zeros = np.linalg.norm(pc[:, 0:3], axis=1)
    r_particle = np.linalg.norm(infos[:, 0:3], axis=1)
    mi = 111
    sum = 0
    b = np.ones(infos.shape[0]).reshape(-1,1)
    infos = np.concatenate((infos,b), axis = 1)
    #print(sum)
    
    np.savetxt(r'F:\code\a.txt',np.concatenate((infos,pc), axis = 0), fmt = "%.2f %.2f %.2f %i")
            
    


