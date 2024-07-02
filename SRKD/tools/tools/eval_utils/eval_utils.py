import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import shutil 
import os 
import random
Visualization = False

if Visualization:
    from tools.demo import show

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    thresh_3 = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    thresh_5 = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[1]
    thresh_7 = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[2]
    disp_dict['recall'] = \
        '(%.2f, %.2f, %.2f)' % (metric['recall_rcnn_%s' % str(thresh_3)]/metric['gt_num']*100, 
                                 metric['recall_rcnn_%s' % str(thresh_5)]/metric['gt_num']*100, 
                                 metric['recall_rcnn_%s' % str(thresh_7)]/metric['gt_num']*100)


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    idx = 0
    want = 153

    #FPS
    time_sum = 0
    cnt = 0
    weatherloss = 0
    Positive = 0
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            


            #pred_dicts, ret_dict, loss = model(batch_dict)
            pred_dicts, ret_dict = model(batch_dict)
        torch.cuda.synchronize()
        end = time.time()
        time_sum += end - start
        cnt += 1
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        for k in range(len(annos)):
            names = annos[k]['name']
            Positive += np.array([names == 'Vehicle']).sum()
            #Positive += np.array([names != 'V1ehicle']).sum()
        print('Precision(0.3,0.5,0.7) = (%.2f,%.2f,%.2f)' % (metric['recall_rcnn_0.3'] / Positive*100,metric['recall_rcnn_0.5'] / Positive*100,metric['recall_rcnn_0.7'] / Positive*100))
        print('FP(0.3,0.5,0.7) = (%d,%d,%d)' % (Positive - metric['recall_rcnn_0.3'],Positive - metric['recall_rcnn_0.5'],Positive - metric['recall_rcnn_0.7']))
        print('Recall(0.3,0.5,0.7) = (%.2f,%.2f,%.2f)' % (metric['recall_rcnn_0.3']/metric['gt_num']*100,metric['recall_rcnn_0.5'] /metric['gt_num']*100,metric['recall_rcnn_0.7']/metric['gt_num']*100))
        print('FN(0.3,0.5,0.7) = (%d,%d,%d)' % (metric['gt_num'] - metric['recall_rcnn_0.3'],metric['gt_num'] - metric['recall_rcnn_0.5'],metric['gt_num'] - metric['recall_rcnn_0.7']))
        vis = False
        if vis:
            box = ret_dict['gt_boxes'].cpu().numpy()
            points = ret_dict['points']

            np.save('/home/hx/models/sun_'+str(cnt), np.array(points.cpu()))
            with open('/home/hx/models/pre_'+str(cnt) + '.pkl', 'wb') as f:
                pickle.dump(annos, f)
            with open('/home/hx/models/box_'+str(cnt) + '.pkl', 'wb') as f:
                pickle.dump(box, f)
        if  vis and 'weather_probability' in batch_dict.keys():
            
            weather_probability = ret_dict['weather_probability']
            weather_labels = ret_dict['weather_labels']
            box = ret_dict['gt_boxes'].cpu().numpy()

            points = ret_dict['points']
            

            xyz = points[:, 1:4]
            x = torch.clamp(torch.floor((xyz[:, 0] + 75.2)/0.1).long(), 0, 1503)
            y = torch.clamp(torch.floor((xyz[:, 1] + 75.2)/0.1).long() , 0, 1503)
            z = torch.clamp(torch.floor((xyz[:, 2] + 2)/0.15).long(), 0, 40)

            weather_labels = weather_labels[z, y, x].int()
            weather_probability = weather_probability[0, :, z, y, x]
            
            weather_pred = xyz.new_zeros(weather_probability.shape).int()
            weather_mask = weather_probability > 0.5
            weather_pred[weather_mask] = 1

            noise = [random.randint(0, weather_labels.shape[0]-1) for i in range(50000)]
            weather_pred[0, noise] = weather_labels[noise]

            points =  torch.cat([xyz, weather_pred.reshape(-1,1)], dim = 1)
            np.save('/home/hx/models/sun_'+str(cnt), np.array(points.cpu()))
            with open('/home/hx/models/pre_'+str(cnt) + '.pkl', 'wb') as f:
                pickle.dump(annos, f)
            with open('/home/hx/models/box_'+str(cnt) + '.pkl', 'wb') as f:
                pickle.dump(box, f)
        
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
    
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    #print('Weatherloss = ', weatherloss)
    #logger.info('Weatherloss = %.3f' %(weatherloss))

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    

    return ret_dict


if __name__ == '__main__':
    pass
