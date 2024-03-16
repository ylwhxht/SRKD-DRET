import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.multiscale_roi_add = [0]#[0, 0, 0, 1]
        self.multiscale_roi_mul = [1, 1.5, 2, 1]
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out * len(self.multiscale_roi_add)

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features
        
    def roi_grid_pool_multiscale(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        multiscale_pooled_features = []
        for idx in range(len(self.multiscale_roi_mul)):
            batch_size = batch_dict['batch_size']
            
            rois = batch_dict['rois'].clone()
            rois[:, :, 3:6] = rois[:, :, 3:6] * self.multiscale_roi_mul[idx] + self.multiscale_roi_add[idx]
            point_coords = batch_dict['point_coords']
            point_features = batch_dict['point_features']
            point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)
            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
                rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            )  # (BxN, 6x6x6, 3)
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

            xyz = point_coords[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            batch_idx = point_coords[:, 0]
            for k in range(batch_size):
                xyz_batch_cnt[k] = (batch_idx == k).sum()

            new_xyz = global_roi_grid_points.view(-1, 3)
            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
            pooled_points, pooled_features = self.roi_grid_pool_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features.contiguous(),
            )  # (M1 + M2 ..., C)

            pooled_features = pooled_features.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            multiscale_pooled_features.append(pooled_features.clone())
        return torch.cat(multiscale_pooled_features, dim = -1) 

    def gtroi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['gt_boxes'][:,:,:7].contiguous()
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        #batch_dict['multiscale_pooled_features'] = pooled_features
        if self.training:
            gt_pooled_features = self.gtroi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

            gt_batch_size_rcnn = gt_pooled_features.shape[0]
            gt_pooled_features = gt_pooled_features.permute(0, 2, 1).\
                contiguous().view(gt_batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

            batch_dict['gt_roifeatures'] = gt_pooled_features
        
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            if 'rain_idx' in batch_dict.keys():
                targets_dict['rain_idx'] = batch_dict['rain_idx']

            if 'weather_labels' in batch_dict.keys():
                targets_dict['weather_labels'] = batch_dict['weather_labels']
                targets_dict['has_label'] = batch_dict['has_label']
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
                )
                targets_dict['batch_box_preds'] = batch_box_preds
                targets_dict['batch_cls_preds'] = batch_cls_preds
                targets_dict['points'] = batch_dict['points']
            if 'weather_labels_points' in batch_dict.keys():
                targets_dict['weather_labels_points'] = batch_dict['weather_labels_points']
                targets_dict['has_label'] = batch_dict['has_label']
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
                )
                targets_dict['batch_box_preds'] = batch_box_preds
                targets_dict['batch_cls_preds'] = batch_cls_preds
                targets_dict['points'] = batch_dict['points']

            self.forward_ret_dict = targets_dict


            batch_dict['cls_preds_normalized'] = False


        return batch_dict
