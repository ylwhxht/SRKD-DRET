import torch
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg
from pcdet.utils import common_utils
from pcdet.models.dense_heads import CenterHead, AnchorHeadTemplate
from ..loss_utils import getKDloss


def forward(model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu, **kwargs):
    optimizer.zero_grad()
    if extra_optim is not None:
        extra_optim.zero_grad()

    load_data_to_gpu(batch)
    ret_dict, tb_dict, disp_dict = model(batch)

    loss = ret_dict['loss']

    batch_sun = batch['data_dict_sun']
    if batch_sun is not None:
        batch_sun['teacher'] = True
        # teacher model works if and only if batch_data has rain data
        with torch.no_grad():
            ret_dict_sun, _, _ = teacher_model(batch_sun)

        KDloss = getKDloss(ret_dict_sun, ret_dict)

        loss = loss + KDloss
        

    loss.backward()
    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
    optimizer.step()
    if extra_optim is not None:
        extra_optim.step()
        
    

    return loss, tb_dict, disp_dict