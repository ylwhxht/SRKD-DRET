from ...utils.spconv_utils import replace_feature, spconv
from functools import partial
import numpy
import torch.nn as nn
import torch



def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

         

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )


        

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        #Save each feature for skip connect
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x_WA(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )


        self.deconv1 = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        self.deconv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        self.logits = spconv.SubMConv3d(16, 1, indice_key="logit", kernel_size=3, stride=1, padding=1,bias=True)

        self.newconv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels+1, 16, 3, padding=1, bias=False, indice_key='newsubm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.newconv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm1'),
        )

        self.newconv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='newspconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm2'),
        )

        self.newconv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='newspconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm3'),
        )
        self.newconv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='newspconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm4'),
        )

        
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.newconv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='newspconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        
        


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        #Save each feature for skip connect
        x_conv1 = self.conv1(x)
        batch_dict['conv1'] = x_conv1
        x_conv2 = self.conv2(x_conv1)
        batch_dict['conv2'] = x_conv2
        
        x_deconv2 = self.deconv2(x_conv2)
        #skip connection
        x_deconv2 = x_deconv2.replace_feature(x_deconv2.features + x_conv1.features)

        x_deconv1 = self.deconv1(x_deconv2)
        
        logits = self.logits(x_deconv1)
        weather_probability = logits.dense()
        
        batch_dict['weather_probability'] = torch.sigmoid(weather_probability)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        voxel_coords_cp = voxel_coords.long()

        weather_features = weather_probability[voxel_coords_cp[:, 0], :, voxel_coords_cp[:, 1], voxel_coords_cp[:, 2], voxel_coords_cp[:, 3]]
        voxel_features =  torch.cat([voxel_features, weather_features], dim = 1)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        newx = self.newconv_input(input_sp_tensor)
        
        #Save each feature for skip connect
        newx_conv1 = self.newconv1(newx)
        batch_dict['conv1'] = newx_conv1
        newx_conv2 = self.newconv2(x_conv1)
        batch_dict['conv2'] = newx_conv2
        newx_conv3 = self.newconv3(newx_conv2)
        batch_dict['conv3'] = newx_conv3
        newx_conv4 = self.newconv4(newx_conv3)
        batch_dict['conv4'] = newx_conv4
        out = self.newconv_out(newx_conv4)
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': newx_conv1,
                'x_conv2': newx_conv2,
                'x_conv3': newx_conv3,
                'x_conv4': newx_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


class VoxelBackBone8x_WAloss(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        
        self.deconv1 = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        self.deconv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        self.deconv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        self.deconv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        self.logits = spconv.SubMConv3d(16, 1, indice_key="logit", kernel_size=3, stride=1, padding=1,bias=True)

        self.newconv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='newsubm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.newconv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm1'),
        )

        self.newconv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='newspconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm2'),
        )

        self.newconv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='newspconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm3'),
        )

        self.newconv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='newspconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='newsubm4'),
        )
        self.attention1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='attention1'),
        )
        self.attention2 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='attention2'),
        )
        self.attention3 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='attention3'),
        )
        self.attention4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='attention4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        
        
    def attention(self, x_conv, newx_conv, attention):
        merx_conv = x_conv.replace_feature(x_conv.features + newx_conv.features)
        merx_conv = attention(merx_conv)
        atten = torch.sigmoid(merx_conv.features)
        x_conv = x_conv.replace_feature(x_conv.features + x_conv.features * atten)
        newx_conv = newx_conv.replace_feature(newx_conv.features + newx_conv.features * atten)
        return x_conv, newx_conv

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        #'batch_size', 'benchmark', 'benchmark_record', 'dense', 'features', 'find_indice_pair', 
        # 'from_dense', 'grid', 'indice_dict', 'indices', 'replace_feature', 'shadow_copy', 
        # 'spatial_shape', 'spatial_size', 'thrust_allocator', 'voxel_num']
    
        newx = self.newconv_input(input_sp_tensor)
        x = self.conv_input(input_sp_tensor)



        #Save each feature for skip connect
        x_conv1 = self.conv1(x)
        newx_conv1 = self.newconv1(newx)
        x_conv1, newx_conv1 = self.attention(x_conv1, newx_conv1, self.attention1)
        batch_dict['conv1'] = newx_conv1
        
        
        x_conv2 = self.conv2(x_conv1)
        newx_conv2 = self.newconv2(newx_conv1)
        x_conv2, newx_conv2 = self.attention(x_conv2, newx_conv2, self.attention2)
        batch_dict['conv2'] = newx_conv2


        x_conv3 = self.conv3(x_conv2)
        newx_conv3 = self.newconv3(newx_conv2)
        x_conv3, newx_conv3 = self.attention(x_conv3, newx_conv3, self.attention3)
        batch_dict['conv3'] = newx_conv3


        x_conv4 = self.conv4(x_conv3)
        newx_conv4 = self.newconv4(newx_conv3)
        x_conv4, newx_conv4 = self.attention(x_conv4, newx_conv4, self.attention4)
        batch_dict['conv4'] = newx_conv4
       
        
        
        
        out = self.conv_out(newx_conv4)

        x_deconv4 = self.deconv4(x_conv4)
        #skip connection
        x_deconv4 = x_deconv4.replace_feature(x_deconv4.features + x_conv3.features)

        x_deconv3 = self.deconv3(x_deconv4)
        #skip connection
        x_deconv3 = x_deconv3.replace_feature(x_deconv3.features + x_conv2.features)

        x_deconv2 = self.deconv2(x_deconv3)
        #skip connection
        x_deconv2 = x_deconv2.replace_feature(x_deconv2.features + x_conv1.features)

        x_deconv1 = self.deconv1(x_deconv2)
        
        logits = self.logits(x_deconv1)
        weather_probability = logits.dense()
        
        batch_dict['weather_probability'] = torch.sigmoid(weather_probability)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        
        
        
        
        #Save each feature for skip connect
        
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': newx_conv1,
                'x_conv2': newx_conv2,
                'x_conv3': newx_conv3,
                'x_conv4': newx_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x_WAShare(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.deconv1 = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        self.deconv2 = block(32, 16, 3, norm_fn=norm_fn, padding=1, indice_key='spconv2', conv_type='inverseconv')
        self.deconv3 = block(64, 32, 3, norm_fn=norm_fn, padding=1, indice_key='spconv3', conv_type='inverseconv')
        
        self.logits = spconv.SubMConv3d(16, 1, indice_key="logit", kernel_size=3, stride=1, padding=1,bias=True)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        #'batch_size', 'benchmark', 'benchmark_record', 'dense', 'features', 'find_indice_pair', 
        # 'from_dense', 'grid', 'indice_dict', 'indices', 'replace_feature', 'shadow_copy', 
        # 'spatial_shape', 'spatial_size', 'thrust_allocator', 'voxel_num']

        x = self.conv_input(input_sp_tensor)



        #Save each feature for skip connect
        x_conv1 = self.conv1(x)
        #x_conv1, newx_conv1 = self.attention(x_conv1, newx_conv1, self.attention1)
        batch_dict['conv1'] = x_conv1
        
        
        x_conv2 = self.conv2(x_conv1)
        #x_conv2, newx_conv2 = self.attention(x_conv2, newx_conv2, self.attention2)
        batch_dict['conv2'] = x_conv2


        x_conv3 = self.conv3(x_conv2)
        #x_conv3, newx_conv3 = self.attention(x_conv3, newx_conv3, self.attention3)
        batch_dict['conv3'] = x_conv3


        x_conv4 = self.conv4(x_conv3)
        #x_conv4, newx_conv4 = self.attention(x_conv4, newx_conv4, self.attention4)
        batch_dict['conv4'] = x_conv4
       
        out = self.conv_out(x_conv4)

        x_deconv3 = self.deconv3(x_conv3)
        #skip connection
        x_deconv3 = x_deconv3.replace_feature(x_deconv3.features + x_conv2.features)


        x_deconv2 = self.deconv2(x_deconv3)
        #skip connection
        x_deconv2 = x_deconv2.replace_feature(x_deconv2.features + x_conv1.features)

        x_deconv1 = self.deconv1(x_deconv2)
        
        logits = self.logits(x_deconv1)
        weather_probability = logits.dense()
        
        batch_dict['weather_probability'] = torch.sigmoid(weather_probability)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        
        
        
        
        #Save each feature for skip connect
        
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
