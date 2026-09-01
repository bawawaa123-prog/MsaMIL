import numpy as np

# 文件路径
coords_path = '/home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/features/4_coords.npy'

# 读取 npy 文件
coords = np.load(coords_path)

# 输出内容
print('坐标 shape:', coords.shape)
print('坐标内容预览:')
print(coords)
