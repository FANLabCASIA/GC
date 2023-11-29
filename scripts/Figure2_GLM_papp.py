from multiprocessing import connection
from sqlite3 import connect
import numpy as np
import scipy.sparse as sps
import nibabel as nib
import statsmodels.api as sm
import gc

hemi = 'R'
data_path = '/public1/home/sch2590/Documents/dyli/HCP_U100'

# load voxel coords in MNI 3mm space
coords_MNI = np.loadtxt(f'{data_path}/tract_space_coords_3mm.txt')

# load MNI 3mm T1 template
MNI_T1 = nib.load(f'{data_path}/MNI152_T1_3mm_brain.nii.gz')

# load hemisphere cerebral roi
hroi = nib.load(f'{data_path}/100307.{hemi}.atlasroi.32k_fs_LR.shape.gii').darrays[0].data
hroi = np.argwhere(hroi>0)

'''
connection_group = np.zeros((hroi.shape[0], coords_MNI.shape[0]))
# loop over 10 npz
for i in range(1,11):
    print(f'***** loading {i}/10 *****')
    connection = sps.load_npz(f'{data_path}/reg{i}_{hemi}.npz')
    connection = connection.toarray()
    connection = connection.astype(np.float32)
    
    connection_group = connection_group + connection
    del connection
    gc.collect()
connection_group = connection_group / 100
connection_group_sp = sps.csr_matrix(connection_group)
sps.save_npz(f'{data_path}/reg_{hemi}.npz', connection_group_sp)
'''

# Calculate voxel contribution 
connection_group = sps.load_npz(f'{data_path}/reg_{hemi}.npz')
connection_group = connection_group.toarray()
# grads = np.load(f'{data_path}/MPC_72fiber_100_embedding_dense_emb_{hemi}_zeros.npy')
grads = np.load(f'{data_path}/MPC_sc_vertex_volume_100_embedding_dense_emb_{hemi}_zeros.npy')
# if hemi == 'R':
#     grads[:, 0] = -1 * grads[:, 0]

result = np.zeros((grads.shape[1], connection_group.shape[1]))
for v in range(connection_group.shape[1]):
    if v==0:
        continue
    if np.sum(connection_group[:,v])==0:
        result[:,v] = result[:,v-1]
    else:
        glm = sm.GLM(connection_group[:,v], grads)
        glm_results = glm.fit()
        result[:,v] = glm_results.params
# np.save(f'{data_path}/voxel_contribution_sc_{hemi}.npy', result)
np.save(f'{data_path}/voxel_contribution_sc_vertex_volume_{hemi}.npy', result)

# loop over first 3 gradients
for g in range(3):
    grad = np.transpose(result[g,:])
    data = np.zeros_like(np.asanyarray(MNI_T1.dataobj)) # (60, 72, 60)
        
    # convert vector to nii to show 
    for i in range(coords_MNI.shape[0]):
        x = int(coords_MNI[i,0])
        y = int(coords_MNI[i,1])
        z = int(coords_MNI[i,2])
        data[x,y,z] = grad[i]
    new_image = nib.Nifti2Image(data, affine=MNI_T1.get_affine())
    # nib.save(new_image, f'{data_path}/voxel_contribution_sc_{hemi}_grad{g}.nii.gz')
    nib.save(new_image, f'{data_path}/voxel_contribution_sc_vertex_volume_{hemi}_grad{g}.nii.gz')
