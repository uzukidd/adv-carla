import torch
import numpy as np
import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
import spconv.pytorch as spconv

features = torch.load("voxel_features.pth").detach().cuda()
indices =  torch.load("voxel_coords.pth").detach().cuda().int() 

print(features)
# 创建稀疏卷积张量
sp_tensor = spconv.SparseConvTensor(features=features, indices=indices, spatial_shape=np.array([41, 1600, 1408]), batch_size=1)
print(sp_tensor.features)
conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(4, 16, 3, padding=1, bias=False, indice_key='subm1'),
        )
# # 前向传播
# output = conv_input(sp_tensor)