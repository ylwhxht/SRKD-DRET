from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x,VoxelBackBone8x_WA,VoxelBackBone8x_WAloss,VoxelBackBone8x_WAShare
from .spconv_unet import UNetV2

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'VoxelBackBone8x_WA': VoxelBackBone8x_WA,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8x_WAloss':VoxelBackBone8x_WAloss,
    'VoxelBackBone8x_WAShare':VoxelBackBone8x_WAShare
}
