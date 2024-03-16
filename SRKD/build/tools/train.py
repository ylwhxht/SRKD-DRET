import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
#改gpu，改batchsize，改getlidar，改includewaymo(包括num_skipped_infos)
from eval_utils import eval_utils
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import math
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, build_teacher_network
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from train_utils.train_utils_KD import train_model_kd
import os
import torch
from tqdm import tqdm
import time



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=6, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='pv_rcnn_sim6000', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=100, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--student_init', action='store_true', default=False, help='')
    #parser.add_argument('--teacher_ckpt', type=str, default='/home/hx/OpenPCDet/output/waymo_models/pv_rcnn/pv_rcnn_Aug/ckpt/checkpoint_epoch_68.pth', help='teacher checkpoint to start from')
    #parser.add_argument('--teacher_ckpt', type=str, default='/home/hx/OpenPCDet/output/waymo_models/pv_rcnn_pairKD/KD_rain+sun_noinit/ckpt/checkpoint_epoch_69.pth', help='teacher checkpoint to start from')
    #parser.add_argument('--teacher_ckpt', type=str, default='/home/hx/OpenPCDet/output/waymo_models/pv_rcnn/pv_rcnn_sim6000/ckpt/checkpoint_epoch_72.pth', help='teacher checkpoint to start from')
    parser.add_argument('--teacher_ckpt', type=str, default='/home/hx/OpenPCDet/output/waymo_models/pv_rcnn_plusplus/fixed_simed/ckpt/checkpoint_epoch_30.pth', help='teacher checkpoint to start from')
    #parser.add_argument('--teacher_ckpt', type=str, default=None, help='teacher checkpoint to start from')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        args.batch_size = args.batch_size 
        # assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        # args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    teacher_model = None
    if args.teacher_ckpt is not None:
        teacher_model = build_teacher_network(cfg, args, train_set, dist_train, logger)
        logger.info('Loading teacher parameters >>>>>>')
        teacher_model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=dist_train, logger=logger)
        
    if args.student_init and args.teacher_ckpt is not None:
        logger.info('Loading student parameters from teacher >>>>>>')
        model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=dist_train, logger=logger)
        

    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
    
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if teacher_model is not None:
        train_model_kd(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            teacher_model=teacher_model
        )
    else:
        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used


if __name__ == '__main__':
    #0, 1, 3, 4, 5, 6
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,5'
    #bash scripts/dist_train.sh 2 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 4 --workers 2 --epochs 50 --extra_tag "pv_rcnn++_fullKD_delincar"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 1 --workers 0 --epochs 80 --extra_tag "pv_rcnn++_fullKD_delincar copy"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn.yaml --batch_size 1 --workers 2 --epochs 80 --extra_tag "pv_rcnn++res"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn.yaml --batch_size 3 --workers 6 --epochs 80 --extra_tag "pv_rcnn_13000sun"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn_WAShare.yaml --batch_size 6 --workers 12 --epochs 60 --extra_tag "pv_rcnn_WAloss_2+2Share_0.1+2.0_sim"
    #bash scripts/dist_train.sh 2   --cfg_file cfgs/waymo_models/pv_rcnn.yaml --batch_size 3 --workers 1 --epochs 80 --extra_tag "pv_rcnn_noisesim6000_re"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn_WAShare.yaml --batch_size 6 --workers 1 --epochs 2 --extra_tag "demo"
    #/usr/bin/python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn_pairKD.yaml --batch_size 3 --workers 6 --epochs 80 --extra_tag "KD_rain+sun_noinit"
    #bash scripts/dist_train.sh 2   --cfg_file cfgs/waymo_models/pv_rcnn_pairKD.yaml --batch_size 3 --workers 1 --epochs 80 --extra_tag "KD_rain+sun" --student_init
    main()
    