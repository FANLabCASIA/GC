import os
import gc
import sys
import time
import h5py
import numpy as np
from numpy.lib.function_base import corrcoef

from sklearn.metrics import pairwise_distances
import scipy.sparse as sps
import scipy as sp
from scipy.sparse.linalg import eigsh, eigs

import argparse
import yaml

########################################################################
#                             config                                   #
config_parser = parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=str, default='0', help='0-3')
parser.add_argument('--hemi', type=str, default='L', help='0-10')

def _parse_args():
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

args, args_text = _parse_args()
print(args)

# config
sub=args.sub
hemi=args.hemi

########################################################################

def compute_diffusion_map_yp(L, alpha=0.5, n_components=None, diffusion_time=0, skip_checks=False, overwrite=False):
    use_sparse = False
    if sps.issparse(L):
        use_sparse = True

    if not skip_checks:
        # Original: from sklearn.manifold.spectral_embedding_ import _graph_is_connected
        from sklearn.manifold._spectral_embedding import _graph_is_connected
        # from sklearn.manifold.spectral_embedding import _graph_is_connected
        if not _graph_is_connected(L):
            raise ValueError('Graph is disconnected')

    ndim = L.shape[0]
    if overwrite:
        L_alpha = L
    else:
        L_alpha = L.copy()

    if alpha > 0:
        # Step 2
        print('Step2...')
        d = np.array(L_alpha.sum(axis=1)).flatten()       # 行相加求和：axis=0, 表示列。axis=1, 表示行. flatten: 返回一个一维数组。
        d_alpha = np.power(d, -alpha)                     # 0.5次方：np.power 求n次方
        if use_sparse:
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
        else:
            L_alpha = d_alpha[:, np.newaxis] * L_alpha    # 插入新维度
            L_alpha = L_alpha * d_alpha[np.newaxis, :]

    # Step 3
    print('Step3...')
    d_alpha = np.power(np.array(L_alpha.sum(axis=1)).flatten(), -1)
    if use_sparse:
        L_alpha.data *= d_alpha[L_alpha.indices]
    else:
        L_alpha = d_alpha[:, np.newaxis] * L_alpha

    M = L_alpha

    # Step 4
    print('Step4...')
    func = eigs
    if n_components is not None:
        lambdas, vectors = func(M, k=n_components + 1)
    else:
        # 求矩阵M的k个特征值和特征向量, 
        # lamadas：ndarray k个特征值的数组. 
        # vector：ndarray k个特征向量的数组. v[:, i]是对应于特征值w [i]的特征向量。
        lambdas, vectors = func(M, k=max(2, int(np.sqrt(ndim))))       
        # lambdas, vectors = calculateKEigs(M)
    del M

    if func == eigsh:
        lambdas = lambdas[::-1]
        vectors = vectors[:, ::-1]
    else:
        lambdas = np.real(lambdas)
        vectors = np.real(vectors)
        lambda_idx = np.argsort(lambdas)[::-1]
        lambdas = lambdas[lambda_idx]
        vectors = vectors[:, lambda_idx]

    print('Step5...')
    return _step_5(lambdas, vectors, ndim, n_components, diffusion_time)

def _step_5(lambdas, vectors, ndim, n_components, diffusion_time):
    """
    This is a helper function for diffusion map computation.

    The lambdas have been sorted in decreasing order.
    The vectors are ordered according to lambdas.

    """
    psi = vectors/vectors[:, [0]]
    diffusion_times = diffusion_time
    if diffusion_time == 0:
        diffusion_times = np.exp(1. -  np.log(1 - lambdas[1:])/np.log(lambdas[1:]))
        lambdas = lambdas[1:] / (1 - lambdas[1:])
    else:
        lambdas = lambdas[1:] ** float(diffusion_time)
    lambda_ratio = lambdas/lambdas[0]
    threshold = max(0.05, lambda_ratio[-1])

    n_components_auto = np.amax(np.nonzero(lambda_ratio > threshold)[0])
    n_components_auto = min(n_components_auto, ndim)
    if n_components is None:
        n_components = n_components_auto
    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]

    result = dict(lambdas=lambdas, vectors=vectors,
                  n_components=n_components, diffusion_time=diffusion_times,
                  n_components_auto=n_components_auto)
    return embedding, result

def calculateKEigs(A, tolerance=0.0001):
        #tol=0表示使用原先的计算精度
        vals, vecs = sp.sparse.linalg.eigs(A, k=10,tol=tolerance,which="SM")
        print("first 3 eigenvalues are %s" % vals)
        k=len(vals)
        nRow,nCol=A.shape[0], A.shape[1]
        for i in range(0,k):
                print("error of lamda is %s " % (np.linalg.norm(A.dot(vecs[:,i])-vals[i]*vecs[:,i])))
        for i in range(0,k):
                v=np.random.rand(nCol)
                #v must be normalization
                v=v/np.linalg.norm(v)
                print("random error is %s " % (np.linalg.norm(A.dot(v)-vals[i]*v)))


# read the sub name
# read the sublist
list_path = './HCP_218.txt'
with open( list_path, 'r' ) as f:
    namelist = [ str( line.strip()) for line in f.readlines() ]
print(f'the number of sub: {len(namelist)}')

list_path_2 = f'/n02dat01/users/dyli/Grad_data/support_data/HCP_U100_list.txt'
with open( list_path_2, 'r' ) as f:
    namelist_2 = [ str( line.strip()) for line in f.readlines() ]
print(len(namelist_2))


# for hemi in ['R']:
#     print(hemi)

#     if hemi == 'L': mpc_l = np.zeros((29696, 29696))
#     if hemi == 'R': mpc_l = np.zeros((29716, 29716))
#     for sub in namelist:
#         print(f'{sub} {hemi} 218all')
#         if sub in namelist_2:
#             mpc = h5py.File(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat')
#             mpc = mpc['MPCi']
#             assert ~np.isnan(mpc).any()
#         else:
#             if os.path.exists(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat'):
#                 mpc = h5py.File(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat')
#                 mpc = mpc['R']
#                 assert ~np.isnan(mpc).any()
#         mpc_l = mpc + mpc_l
#     np.savetxt(f'/n01dat01/dyli/data4n02/HCP_218_mpc_{hemi}.txt', mpc_l)

#     sparse_idx=90
#     print(sparse_idx)
#     # 稀疏
#     dist_mat = mpc_l.astype(np.float32)
#     perc_L = np.array([np.percentile(x, sparse_idx) for x in dist_mat])
#     for i in range(dist_mat.shape[0]):
#         dist_mat[i, dist_mat[i,:] < perc_L[i]] = 0

#     # 去除负连接
#     dist_mat[dist_mat<0] = 0

#     # 计算相似性矩阵的相似性矩阵
#     aff = 1 - pairwise_distances(dist_mat, metric = 'cosine')
#     emb, res = compute_diffusion_map_yp(aff, alpha = 0.5)  
#     lambdas = res['lambdas']
#     np.save(f'/n01dat01/dyli/data4n02/HCP_218_sparse90_embedding_dense_lambda_{hemi}.npy', lambdas)
#     np.save(f'/n01dat01/dyli/data4n02/HCP_218_sparse90_embedding_dense_emb_{hemi}.npy', emb)

for hemi in ['R']:
    print(hemi)

    if hemi == 'L': mpc_l = np.zeros((29696, 29696))
    if hemi == 'R': mpc_l = np.zeros((29716, 29716))
    for sub in namelist:
        print(f'{sub} {hemi} 218 run2')
        if sub in namelist_2:
            pass
        elif os.path.exists(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat'):
            mpc = h5py.File(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat')
            mpc = mpc['R']
            assert ~np.isnan(mpc).any()
            mpc_l = mpc + mpc_l
        else:
            print(f'{sub} has no MPC!')
    np.savetxt(f'/n01dat01/dyli/data4n02/HCP_218_run2_mpc_{hemi}.txt', mpc_l)

    sparse_idx=90
    print(sparse_idx)
    # 稀疏
    dist_mat = mpc_l.astype(np.float32)
    perc_L = np.array([np.percentile(x, sparse_idx) for x in dist_mat])
    for i in range(dist_mat.shape[0]):
        dist_mat[i, dist_mat[i,:] < perc_L[i]] = 0

    # 去除负连接
    dist_mat[dist_mat<0] = 0

    # 计算相似性矩阵的相似性矩阵
    aff = 1 - pairwise_distances(dist_mat, metric = 'cosine')
    emb, res = compute_diffusion_map_yp(aff, alpha = 0.5)  
    lambdas = res['lambdas']
    np.save(f'/n01dat01/dyli/data4n02/HCP_218_run2_sparse90_embedding_dense_lambda_{hemi}.npy', lambdas)
    np.save(f'/n01dat01/dyli/data4n02/HCP_218_run2_sparse90_embedding_dense_emb_{hemi}.npy', emb)


# # read the sub name
# # read the sublist
# list_path = './HCP_twins.txt'
# with open( list_path, 'r' ) as f:
#     namelist = [ str( line.strip()) for line in f.readlines() ]
# print(f'the number of sub: {len(namelist)}')

# for hemi in ['L', 'R']:
#     print(hemi)

#     if hemi == 'L': mpc_l = np.zeros((29696, 29696))
#     if hemi == 'R': mpc_l = np.zeros((29716, 29716))
#     for sub in namelist:
#         print(f'twin {sub} {hemi}')
#         if os.path.exists(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat'):
#             mpc = h5py.File(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat')
#             mpc = mpc['R']
#             assert ~np.isnan(mpc).any()
#             mpc_l = mpc + mpc_l
#         else:
#             print(f'{sub} has no MPC!')
#     np.savetxt(f'/n01dat01/dyli/data4n02/HCP_twins_mpc_{hemi}.txt', mpc_l)

#     sparse_idx=90
#     print(sparse_idx)
#     # 稀疏
#     dist_mat = mpc_l.astype(np.float32)
#     perc_L = np.array([np.percentile(x, sparse_idx) for x in dist_mat])
#     for i in range(dist_mat.shape[0]):
#         dist_mat[i, dist_mat[i,:] < perc_L[i]] = 0

#     # 去除负连接
#     dist_mat[dist_mat<0] = 0

#     # 计算相似性矩阵的相似性矩阵
#     aff = 1 - pairwise_distances(dist_mat, metric = 'cosine')
#     emb, res = compute_diffusion_map_yp(aff, alpha = 0.5)  
#     lambdas = res['lambdas']
#     np.save(f'/n01dat01/dyli/data4n02/HCP_twins_sparse90_embedding_dense_lambda_{hemi}.npy', lambdas)
#     np.save(f'/n01dat01/dyli/data4n02/HCP_twins_sparse90_embedding_dense_emb_{hemi}.npy', emb)