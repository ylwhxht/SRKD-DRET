import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import time
from . import box_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
#from chamferdist import ChamferDistance
class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)
def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point1, num_features = array1.shape
    num_point2, num_features = array2.shape
    expanded_array1 = np.tile(array1, (num_point2, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point1, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1 - expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point1, num_point2))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    batch_size = len(array1)
    dist = torch.zeros(batch_size)
    chamferDist = ChamferDistance()
    for i in range(batch_size):
        if array1[i].shape[0] == 0 or array2[i].shape[0] == 0:
            dist[i] = 1e9
            continue
        #that because it's very simility, unnecessary to calculate to save the time(according to point weight)
        if (array1[i].shape[0] - array2[i].shape[0]) / array2[i].shape[0] < 0.3333:
            dist[i] = 0
            continue
        pc1 = torch.tensor(array1[i]).unsqueeze(0)
        pc2 = torch.tensor(array2[i]).unsqueeze(0)
        dist[i] = chamferDist(pc1, pc2, bidirectional=True)
    return (1 - torch.tanh(dist))

def get_InsRoIFeature_KD(batch_dict: dict, batch_dict_rain: dict, pairloss):
    """
    Args:
        feature_sun: roi feature from sun: float Tensor.
        feature_rain: roi feature from rain: float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    LossFun = torch.nn.SmoothL1Loss(reduction='none')
    feature_sun = batch_dict['gt_roifeatures']
    feature_sun = feature_sun.view(feature_sun.size(0), -1)
    feature_rain = batch_dict_rain['gt_roifeatures']
    feature_rain = feature_rain.view(feature_rain.size(0), -1)
    #get rain_simulation batch
    rain_idx = batch_dict_rain['rain_idx'].long()

    #get rain_simulation gt_roifeatures under ans rain
    feadim = feature_sun.shape[1]
    feature_rain = feature_rain.reshape(batch_dict_rain['batch_size'], -1, feadim)
    feature_rain = feature_rain[rain_idx,...]
    feature_rain = feature_rain.reshape(-1, feadim)
    points_obj = [] # sun
    points_obj_rain = [] #rain
    
    box_idx_num = np.zeros((batch_dict['batch_size'],batch_dict['gt_boxes'].shape[1]),dtype = np.int16)
    box_idx_rain_num = np.zeros((batch_dict['batch_size'],batch_dict['gt_boxes'].shape[1]),dtype = np.int16)
    
    for bs_idx in range(batch_dict['batch_size']):
        #get  points in gt_boxes under rain and sun
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!notice!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: 
        #bs_idx is batch_index in sun batches, however rain_idx[bs_idx] only then is matched batch_index in rain batches
        points = batch_dict['points'].cpu().numpy()
        points = points[points[:,0]==bs_idx][:, 1:4]

        gt_boxes = batch_dict['gt_boxes'][bs_idx, :, 0:7].cpu().numpy()
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()
        
        points_rain = batch_dict_rain['points'].cpu().numpy()
        points_rain = points_rain[points_rain[:,0] == int(rain_idx[bs_idx])][:, 1:4]
        
        box_idxs_of_pts_rain = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points_rain).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()
        
        #calculate the number of points in each object under rain and sun
        
        for i in range(gt_boxes.shape[0]):
            box_idx_num[bs_idx, i] = (box_idxs_of_pts == i).sum()
            box_idx_rain_num[bs_idx, i] = (box_idxs_of_pts_rain == i).sum()
            points_obj.append(points[box_idxs_of_pts == i])
            points_obj_rain.append(points_rain[box_idxs_of_pts_rain == i])
            

    assert feature_sun.shape == feature_rain.shape
    
    
    eps = 0.0001
    
    min_box_num = np.array(box_idx_num, dtype = int)
    # shape = chamfer_distance_numpy(points_obj,points_obj_rain).cuda()

    min_box_num[box_idx_num > box_idx_rain_num] = box_idx_rain_num[box_idx_num > box_idx_rain_num]
    weight = torch.tanh(torch.tensor(min_box_num / np.abs(box_idx_num - box_idx_rain_num + eps))).reshape(-1).cuda()
    
    
    # for i in range(len(dist)):
    #     print("{%d, %d} : Shape_Weight = %.4f, Density_Weight = %.4f, Final_Weight = %.4f"%(box_idx_num.reshape(-1)[i].item(), box_idx_rain_num.reshape(-1)[i].item(), dist[i].item(),  weight[i].item(), (dist[i] * weight[i]).item()))
    distance = LossFun(feature_sun, feature_rain)
    kd_feature_loss = (torch.mean(distance, dim = 1) * weight).mean() * pairloss
    
    #kd_feature_loss = (torch.mean(distance, dim = 1)).mean() * pairloss
    
    # visual = torch.mean(distance, dim = [1,2,3,4])
    # np.save('/home/hx/models/pc/sun_'+str(kd_feature_loss.item()), np.array(batch_dict['points'].cpu()))
    # np.save('/home/hx/models/pc/rain_'+str(kd_feature_loss.item()),np.array(batch_dict_rain['points'].cpu()))
    # box = np.concatenate([batch_dict_rain['gt_boxes'].reshape(-1,8).cpu().numpy(), visual.detach().cpu().numpy().reshape(-1,1)],axis=1)

    # with open('/home/hx/models/box/box_'+str(kd_feature_loss.item()) + '.pkl', 'wb') as f:
    #     pickle.dump(box, f)
    return kd_feature_loss

def get_MulRoIFeature_KD(batch_dict: dict, batch_dict_rain: dict, multi_roi_loss):
    feature_sun = batch_dict['multiscale_pooled_features']
    if 'rain_idx' in batch_dict_rain.keys():
        shape = batch_dict['multiscale_pooled_features'].shape
        feature_sun = feature_sun.reshape(batch_dict['batch_size'], -1, shape[1], shape[2], shape[3], shape[4])
        feature_sun = feature_sun[batch_dict_rain['rain_idx'].long(),...]
        feature_sun = feature_sun.reshape(-1, shape[1], shape[2], shape[3], shape[4])
    feature_rain = batch_dict_rain['multiscale_pooled_features']
    LossFun = torch.nn.SmoothL1Loss(reduction='none')
    distance = LossFun(feature_rain, feature_sun)
    kd_feature_loss = (torch.mean(distance, dim = [1,2,3,4])).mean() * multi_roi_loss
    return kd_feature_loss

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def add_sin_difference(boxes1, boxes2, dim=6):
    assert dim != -1
    rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
    rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
    boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
    return boxes1, boxes2

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def get_Logit_KD(batch_dict: dict, batch_dict_rain: dict, clsweight, regweight):
    #cite SparseKD
    ClsLossFun = torch.nn.MSELoss(reduction = 'none')
    if 'cls_preds' in batch_dict.keys():
        #anchor based 
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        RegLossFun = WeightedSmoothL1Loss(code_weights = code_weights)
        
        
        if 'rain_idx' in batch_dict_rain.keys():
            cls_stu = batch_dict_rain['cls_preds'].clone()
            box_stu = batch_dict_rain['box_preds'].clone()
            cls_stu = cls_stu[batch_dict_rain['rain_idx'].long(), ...]
            box_stu = box_stu[batch_dict_rain['rain_idx'].long(), ...]

            
        cls_tea = batch_dict['cls_preds']
        box_tea = batch_dict['box_preds']

        #cls KD
        assert cls_stu.shape == cls_tea.shape
        cls_stu = sigmoid(cls_stu)
        cls_tea = sigmoid(cls_tea)
        kd_cls_loss = ClsLossFun(cls_stu, cls_tea).mean() * clsweight

        #reg KD
        num_anchors_per_location = batch_dict['num_anchors_per_location']
        box_cls_labels = batch_dict['box_cls_labels']
        batch_size = int(box_stu.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        box_stu = box_stu.view(batch_size, -1, box_stu.shape[-1] // num_anchors_per_location)
        box_tea = box_tea.view(batch_size, -1, box_tea.shape[-1] // num_anchors_per_location)

        # sin(a - b) = sinacosb-cosasinb
        box_stu_sin, box_tea_sin = add_sin_difference(box_stu, box_tea)
        kd_reg_loss = RegLossFun(box_stu_sin, box_tea_sin, weights=reg_weights)  # [N, M]
        kd_reg_loss = kd_reg_loss.sum() / batch_size * regweight 
        kd_logit_loss = kd_reg_loss + kd_cls_loss
        
    else:
        #anchor free centerbased
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        target_dicts = batch_dict_rain['target_dicts']
        HEAD_ORDER = [ 'center', 'center_z', 'dim', 'rot' ]
        pred_stu = batch_dict_rain['pred_dicts']
        loc_loss = 0
        hm_loss = 0
        if 'rain_idx' in batch_dict_rain.keys():
            rain_idx = []
            for idx in batch_dict_rain['rain_idx']:
                rain_idx.append(idx.int().item())
        else :
            return 0
        # if 'rain_idx' in batch_dict_rain.keys():
        #     pred_stu_tmp = batch_dict_rain['pred_dicts']
        #     for idx, cur_pred_stu in enumerate(pred_stu_tmp):
        #         print(idx)
        #     for idx in batch_dict_rain['rain_idx']:
        #         pred_stu.append(pred_stu_tmp[idx.int()])
        
        hm_loss_func = torch.nn.MSELoss(reduction = 'none')
        reg_loss_func = RegLossCenterNet()
        pred_tea = batch_dict['pred_dicts']
        kd_logit_loss = 0
        
        for idx in range(len(pred_stu)):
            pred_dict_stu = pred_stu[idx]
            pred_dict_tea = pred_tea[idx]
            pred_dict_stu_hm = pred_dict_stu['hm'][rain_idx]
            pred_dict_tea_hm = sigmoid(pred_dict_tea['hm'])

            hm_loss_raw = ClsLossFun(pred_dict_stu_hm, pred_dict_tea_hm).mean()
            mask = (torch.max(pred_dict_tea_hm, dim=1)[0] > 0.5).float()
            hm_loss += (hm_loss_raw * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6) * clsweight

            pred_boxes_stu = torch.cat([pred_dict_stu[head_name][rain_idx] for head_name in HEAD_ORDER], dim=1)
            pred_boxes_tea = torch.cat([pred_dict_tea[head_name] for head_name in HEAD_ORDER], dim=1)

            reg_loss = reg_loss_func(
                pred_boxes_stu, target_dicts['masks'][idx][rain_idx], target_dicts['inds'][idx][rain_idx], pred_boxes_tea, KD = True
            )
            loc_loss_raw = (reg_loss * reg_loss.new_tensor(code_weights)).sum()
            loc_loss += loc_loss_raw * regweight
        # print("hm_loss : ", hm_loss)
        # print("loc_loss : ", loc_loss)
        kd_logit_loss = (hm_loss + loc_loss) / len(pred_stu)

    return kd_logit_loss


def get_Sim_KD(batch_dict: dict, batch_dict_rain: dict, weight):
    bev_stu = batch_dict_rain['spatial_features_2d'].clone()
    bev_tea = batch_dict['spatial_features_2d'].clone()
    gt_box = batch_dict['gt_boxes'].clone()
    box_bev = []
    for bs in range(gt_box.shape[0]):
        box2d = box_utils.boxes3d_lidar_to_aligned_bev_boxes(gt_box[bs])
        #map:[b, c, y, x]
        gt_box[:, :, :4] = box2d


    bev_fg_mask = compute_fg_mask(gt_box[:, :, :4], shape = bev_stu.shape, downsample_factor = 8, device = bev_stu.device)

    similarity_bs = torch.zeros(bev_tea.shape[0], dtype=torch.float64, device=bev_stu.device)
    for bs in range(bev_tea.shape[0]):
        bev_stu_bs = bev_stu[bs][bev_fg_mask[bs]].view(1, -1)
        bev_tea_bs = bev_tea[bs][bev_fg_mask[bs]].view(1, -1)
        bev_stu_bs = F.normalize(bev_stu_bs)
        bev_tea_bs = F.normalize(bev_tea_bs)
        similarity_bs[bs] = torch.cosine_similarity(bev_stu_bs, bev_tea_bs, dim = 1)
        similarity_bs[bs] = 1 - ((similarity_bs[bs] + 1) / 2)#[-1, 1]->[0,1]

    
    
    kd_sim_loss = similarity_bs.mean() * weight

    return kd_sim_loss

def getKDloss(ret_dict, ret_dict_rain):
    InsRoIFeatureKD_loss = get_InsRoIFeature_KD(ret_dict, ret_dict_rain, 4.0)
    print("InsRoIFeatureKD_loss", InsRoIFeatureKD_loss)
    LogitKD_loss = get_Logit_KD(ret_dict, ret_dict_rain, 1, 0.02)
    print("LogitKD_loss : ", LogitKD_loss)
    return InsRoIFeatureKD_loss, LogitKD_loss
    return -1, LogitKD_loss


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, :, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None, KD=False):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
            if KD:
                target = _transpose_and_gather_feat(target, ind)
        loss = _reg_loss(pred, target, mask)
        return loss