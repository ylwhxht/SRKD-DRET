from .detector3d_template import Detector3DTemplate


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            if not 'teacher' in batch_dict.keys():
                loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
                
                ret_dict = {
                    'loss': loss
                }
            else:
                tb_dict = None
                disp_dict = None
                ret_dict = {
                }
            if 'rain_idx' in batch_dict.keys() or 'teacher' in batch_dict.keys():
                keyinfos = ['gt_roifeatures','batch_size', 'rain_idx', 'gt_boxes', 'points','cls_preds','pred_dicts', 'box_preds', 'num_anchors_per_location', 'box_cls_labels','target_dicts']
                for key in keyinfos:
                    if key in batch_dict.keys():
                        ret_dict[key] = batch_dict[key]
                
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if False and 'has_label' in batch_dict.keys():
            
            loss_rcnn, loss_weather, tb_dict = self.roi_head.get_loss(tb_dict)
            #loss_rcnn, loss_weather, tb_dict = self.roi_head.get_loss(tb_dict
            loss = loss_rpn  + loss_rcnn + loss_weather
        else:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            #loss_rcnn, loss_weather, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_rpn  + loss_rcnn

        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict
