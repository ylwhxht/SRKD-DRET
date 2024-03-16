from .detector3d_template import Detector3DTemplate
import torch
import numpy as np 
import random
import pickle
class PVRCNN_WAloss(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            keyinfos = ['spatial_features_2d','gt_roifeatures', 'batch_size', 'rain_idx', 'gt_boxes', 'points', 'multiscale_pooled_features', 'cls_preds', 'box_preds', 'num_anchors_per_location', 'box_cls_labels']
            for key in keyinfos:
                if key in batch_dict.keys():
                    ret_dict[key] = batch_dict[key]
            return ret_dict, tb_dict, disp_dict
        else:
            #loss_weather, _ = self.roi_head.get_weather_loss()
            

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
        
            if 'weather_probability' in batch_dict.keys():
                recall_dicts['weather_probability'] = batch_dict['weather_probability']
            if 'weather_labels' in batch_dict.keys():
                recall_dicts['weather_labels'] = batch_dict['weather_labels']
            if 'points' in batch_dict.keys():
                recall_dicts['points'] = batch_dict['points']
            if 'gt_boxes' in batch_dict.keys():
                recall_dicts['gt_boxes'] = batch_dict['gt_boxes']
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        if 'has_label' in batch_dict.keys():
            #loss_rcnn, loss_weather, loss_consistency, tb_dict = self.roi_head.get_loss(tb_dict)
            loss_rcnn, loss_weather, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_rpn + loss_point + loss_rcnn + loss_weather# + loss_consistency
        else:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            #loss_rcnn, loss_weather, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
