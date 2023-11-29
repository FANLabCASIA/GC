import os
import sys
import numpy as np
import scipy.sparse as sps
import scipy as sp

import argparse
import yaml


########################################################################
#                             config                                   #
config_parser = parser = argparse.ArgumentParser()
parser.add_argument('--subi', type=int, default='0', help='0-3')
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
subi=args.subi
hemi=args.hemi

########################################################################

def zero_row_check(connection):
    # 检查 connection 是否有全 0 行
    for ii in range(connection.shape[0]):
        row = connection[ii, :]
        xx = np.unique(row)
        if len(xx) == 1:
            connection[ii, :] = connection[ii-1, :]
    for ii in range(connection.shape[0]):
        row = connection[ii, :]
        xx = np.unique(row)
        if len(xx) == 1:
            print(ii)
            sys.exit(0)
    return connection

# read the sub name
name_path = './HCP_twins.txt'
with open( name_path, 'r' ) as f:
    namelist = [ str( line.strip()) for line in f.readlines() ]

# for subi,sub in enumerate(NC_namelist):
sub = namelist[subi]
print(subi,sub, hemi)

if not os.path.exists(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}'):
    os.mkdir(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}')

if not os.path.exists(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/{sub}_{hemi}_sc_MPC.mat'):
    print(f'{sub} not has MPC')
    sc = sps.load_npz(f'/n04dat01/atlas_group/lma/HCP_S1200_individual_MSM_atlas/{sub}/{sub}_{hemi}_probtrackx_omatrix2/fdt_matrix2.npz')
    sc = sc.toarray()
    sc = zero_row_check(sc)
    np.savetxt(f'/n01dat01/dyli/data4n02/HCP_twin/{sub}/vertex_volume_sc_{hemi}.txt', sc)
    subi = int(subi+1)
    os.system(f'matlab -nodisplay -nosplash -r \"MPC_calculate_HCP_R({subi});exit;\"')
else:
    print(f'{sub} has MPC!')