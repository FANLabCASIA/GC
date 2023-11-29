import os
import gc
import sys
import time
import h5py
import torch
import numpy as np
from numpy.lib.function_base import corrcoef
import nibabel as nib
# from nilearn import image, surface, plotting, datasets

from sklearn.metrics import pairwise_distances
import scipy.sparse as sps
import scipy as sp
from scipy.sparse.linalg import eigsh, eigs
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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


########################################################################
# 存储gii
########################################################################

def saveSurf( template, data, path, save_name ):
    template.remove_gifti_data_array(0)
    template.add_gifti_data_array( nib.gifti.gifti.GiftiDataArray( np.array( data, np.float32 )))
    nib.loadsave.save( template, os.path.join( path, save_name ))
'''
hemi='R'
dirc_R = f'/n02dat01/users/dyli/Atlas/metric_index_{hemi}.txt'
select_ind_R = np.loadtxt( dirc_R ).astype(int)
emb = np.load(f'/n02dat01/users/dyli/Plot/MPC_sc_embedding_dense_emb_{hemi}.npy')
for gg in range(4):
    Grad_data = np.zeros(32492)
    Grad_data[select_ind_R] = emb[:,gg]
    pipeline_path = '/n02dat01/users/dyli/Plot'
    template = nib.loadsave.load(f'/n14dat01/lma/data/fsaverage_LR32k/fsaverage.{hemi}.BN_Atlas.32k_fs_LR.label.gii')
    save_path = f'/n02dat01/users/dyli/Plot/Grad_results/MPC_sc_Grad{gg}.{hemi}.func.gii'
    saveSurf(template, Grad_data, pipeline_path, save_path)
'''

########################################################################
# 颜色相关
########################################################################
'''
# 生成指定 colorbar 的 RGB 颜色和 Hex 颜色
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

color_num = 10

norm1 = mpl.colors.LogNorm()
norm2 = mpl.colors.TwoSlopeNorm(0.4)

sm1 = mpl.cm.ScalarMappable(norm1, 'ocean')
x = np.arange(1,int(color_num+1),1)
rgb_color = 255 * sm1.to_rgba(x)
rgb_color = rgb_color.astype(np.int32)
rgb_color = rgb_color[:, 0:3]

def RGB_to_Hex(RGB):
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

hex_color = []
for rgbi in range(color_num):
    rgb_ = np.squeeze(rgb_color[rgbi,:])
    hex_color.append(RGB_to_Hex(rgb_))

# 绘制自定义的 colorbar
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colors = ['#005000', '#007F00','#FFFFFF', '#00679A', '#000C4D']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# cmap = mpl.cm.viridis
cmap = cmap1
bounds = [-1, 2, 5, 7, 12, 15]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb2.set_label("Discrete intervals with extend='both' keyword")
fig.show()
'''

def RGB_to_Hex(RGB):
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def Hex_to_RGB(hex):
    hex = hex.strip('#')
    r = int(hex[0:2],16)
    g = int(hex[2:4],16)
    b = int(hex[4:6], 16)
    # rgb = str(r)+','+str(g)+','+str(b)
    rgb = [r,g,b]
    # print(rgb)
    return rgb

########################################################################
# 绘制 brain surface
########################################################################

def plot_surface_my(emb, hemi, cmap1='coolwarm', name=''):
    '''
    emb:   the gradient data;       exp: emb = np.load('/n02dat01/users/dyli/Grad_data/sc/MPC_Grad_results/MPC_sc_vertex_volume_100_embedding_dense_emb_L_zeros.npy')
    hemi:  the hemi of the brain;   exp: 'L' , 'R'
    cmap1: the colorbar
    name:  the name of Gradient
    '''
    from nilearn import plotting

    if hemi == 'L':
        zz_L = np.zeros(32492)
        zz_L[:] = np.nan
        dirc_L = '/n02dat01/users/dyli/Atlas/metric_index_L.txt'
        select_ind_L = np.loadtxt( dirc_L ).astype(int)

        zz_L[select_ind_L] = np.squeeze(emb)

        fig_L = plotting.plot_surf_stat_map('/n02dat01/users/dyli/MNI152/MNINonLinear/fsaverage_LR32k/MNI152.L.inflated.32k_fs_LR.surf.gii', 
            zz_L, 
            hemi='left',
            title=f'{name}: left', 
            colorbar=True, 
            cmap=cmap1)
        fig_L.show()

        fig_L = plotting.plot_surf_stat_map('/n02dat01/users/dyli/MNI152/MNINonLinear/fsaverage_LR32k/MNI152.L.inflated.32k_fs_LR.surf.gii', 
            zz_L, 
            hemi='left',
            view='medial',
            title=f'{name}: left', 
            colorbar=True, 
            cmap=cmap1)
        fig_L.show()

    elif hemi == 'R':
        zz_R = np.zeros(32492)
        zz_R[:] = np.nan
        dirc_R = '/n02dat01/users/dyli/Atlas/metric_index_R.txt'
        select_ind_R = np.loadtxt( dirc_R ).astype(int)

        zz_R[select_ind_R] = np.squeeze(emb)

        fig_R = plotting.plot_surf_stat_map('/n02dat01/users/dyli/MNI152/MNINonLinear/fsaverage_LR32k/MNI152.R.inflated.32k_fs_LR.surf.gii', 
            zz_R, 
            hemi='right',
            title=f'{name}: right', 
            colorbar=True, 
            cmap=cmap1)
        fig_R.show()

        fig_R = plotting.plot_surf_stat_map('/n02dat01/users/dyli/MNI152/MNINonLinear/fsaverage_LR32k/MNI152.R.inflated.32k_fs_LR.surf.gii', 
            zz_R, 
            hemi='right',
            view='medial',
            title=f'{name}: right', 
            colorbar=True, 
            cmap=cmap1)
        fig_R.show()


########################################################################
# 将左右脑放在一起
########################################################################
def whole_brain(data_L: np.ndarray, data_R: np.ndarray) -> np.ndarray:
    '''
    data_L : (32492, )
    data_R : (32492, )
    '''
    dirc_L = f'/n02dat01/users/dyli/Atlas/metric_index_L.txt'
    select_ind_L = np.loadtxt( dirc_L ).astype(int)
    dirc_R = f'/n02dat01/users/dyli/Atlas/metric_index_R.txt'
    select_ind_R = np.loadtxt( dirc_R ).astype(int)

    data = np.zeros(59412)
    data[0:29696] = data_L[select_ind_L]
    data[29696:59412] = data_L[select_ind_R]
    return data

########################################################################
# 绘制线性回归图
########################################################################
def linear_regplot(x:np.ndarray, y:np.ndarray, x_name:str, y_name:str, save_path:str):
    import pandas as pd
    import seaborn as sns
    '''
    x: (n, )
    y: (n, )
    '''
    x = np.squeeze(x)
    y = np.squeeze(y)

    # 将自变量和因变量放在一起
    y_ = np.zeros((x.shape[0],2))
    y_[:,0] = y # 纵坐标
    y_[:,1] = x # 横坐标
    y_df = pd.DataFrame(y_, columns=[y_name, x_name]) # columns=[纵坐标名字，横坐标名字]

    # 绘制回归图
    plt.figure(figsize=(10, 10))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    sns.regplot(x=x_name, y=y_name, data=y_df, 
                scatter_kws={"color": '#6581EB', 's':1},
                line_kws={"color": "#B30426"})
    plt.ylabel(y_name,fontsize=25)
    plt.xlabel(x_name,fontsize=25)
    # sns.jointplot(x="contri", y="FA", data=y_df, kind="reg")
    plt.savefig(save_path, transparent = True, bbox_inches = 'tight',dpi = 700)
    plt.show()