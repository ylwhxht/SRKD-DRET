from .detector3d_template import Detector3DTemplate

class PVRCNN_PAIR(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.pairloss = self.model_cfg.PAIR_INPUT_WEIGHT

    def forward(self, batch_dict):
        batch_dict_rain = batch_dict['data_dict_rain']
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            self.batch_dict = batch_dict
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }

            if batch_dict_rain is not None:
                for cur_module in self.module_list:
                    batch_dict_rain = cur_module(batch_dict_rain)
                self.batch_dict_rain = batch_dict_rain
                loss_rain, tb_dict_rain, disp_dict_rain = self.get_training_loss(True)
                ret_dict['loss_rain'] = loss_rain
            
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,cal_distance = False):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_point + loss_rcnn
        if cal_distance:
            distance_loss = get_Feature_Distance(self.batch_dict,self.batch_dict_rain,self.pairloss)
            loss += distance_loss
        return loss, tb_dict, disp_dict
