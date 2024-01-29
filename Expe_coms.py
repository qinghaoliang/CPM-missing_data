####################################################################
########### Experiment CPM with missing connectomes ################
####################################################################

import h5py
import numpy as np
import scipy.io as sio
import argparse
from mats2edges import mats2edges
from cpm_imp import cpm_imp, cpm_imp_clf
from cpm_imp_coms import cpm_imp_coms, cpm_imp_coms_clf
from cpm_mc import cpm_mc, cpm_mc_clf
from cpm import cpm, cpm_clf

parser = argparse.ArgumentParser(description='real missing data')
parser.add_argument('-sd',type=int,help="seed")
args = parser.parse_args()
seed = args.sd

# load the data set
# all_mats: connectomes, should be rendered in shape (m,m,n,k)
# m is the number of nodes, n is the number of subject, k the number of tasks
# all_behavs: phenotypes, (n, k)

print("seed = %d" % seed)
all_mats = h5py.File("./data/all_mats_part.mat", 'r')
all_mats = all_mats['all_mats_part']
all_mats = np.array(all_mats)
all_mats = np.transpose(all_mats, (3, 2, 1, 0))
all_behavs = sio.loadmat("./data/behavs.mat")
all_behavs = all_behavs['behavs']

# Find out which connectomes are missing 
print(np.shape(all_mats))
m_miss = all_mats[0, 0, :, :]
m_miss = np.isnan(m_miss)
m_miss = m_miss.astype(int)
sub_miss = np.sum(m_miss, axis=1)

# Separate the participants into ones with missing data and ones don't
sub = (sub_miss > 0)
all_mats_ic = all_mats[:,:,sub,:]
all_edges_ic = mats2edges(all_mats_ic)
all_behavs_ic = all_behavs[sub,:]
all_mats_cp = all_mats[:,:,sub_miss==0,:]
all_edges_cp = mats2edges(all_mats_cp)
all_behavs_cp = all_behavs[sub_miss==0,:]

# try different imputation methods for sex classification
sex = np.zeros(6)
all_sex_cp = all_behavs_cp[:,1].astype(float)
all_sex_cp = all_sex_cp.reshape(-1)
all_sex_ic = all_behavs_ic[:,1].astype(float)
all_sex_ic = all_sex_ic.reshape(-1)
sex[0] = cpm_clf(all_edges_cp, all_sex_cp, seed)
sex[1] = cpm_imp_clf(all_edges_ic, all_sex_ic, all_edges_cp, all_sex_cp, seed, "const")
sex[2] = cpm_imp_clf(all_edges_ic, all_sex_ic, all_edges_cp, all_sex_cp, seed, "mean")
sex[3] = cpm_imp_coms_clf(all_mats_ic, all_sex_ic, all_edges_cp, all_sex_cp, seed)
sex[4] = cpm_imp_clf(all_edges_ic, all_sex_ic, all_edges_cp, all_sex_cp, seed)
sex[5] = cpm_mc_clf(all_edges_ic, all_sex_ic, all_edges_cp, all_sex_cp, seed)

filename = "../results/cnp_sex_" + str(seed) + ".npy"
np.save(filename, sex)

# try different imputation methods for age prediction
age = np.zeros((6,2))
all_age_cp = all_behavs_cp[:,0].astype(float)
all_age_cp = all_age_cp.reshape(-1)
all_age_ic = all_behavs_ic[:,0].astype(float)
all_age_ic = all_age_ic.reshape(-1)
age[0,0], age[0,1] = cpm(all_edges_cp, all_age_cp, seed)
age[1,0], age[1,1] = cpm_imp(all_edges_ic, all_age_ic, all_edges_cp, all_age_cp, seed, "const")
age[2,0], age[2,1] = cpm_imp(all_edges_ic, all_age_ic, all_edges_cp, all_age_cp, seed, "mean")
age[3,0], age[3,1] = cpm_imp_coms(all_mats_ic, all_age_ic, all_edges_cp, all_age_cp, seed)
age[4,0], age[4,1] = cpm_imp(all_edges_ic, all_age_ic, all_edges_cp, all_age_cp, seed)
age[5,0], age[5,1] = cpm_mc(all_edges_ic, all_age_ic, all_edges_cp, all_age_cp, seed)

filename = "../results/cnp_age_" + str(seed) + ".npy"
np.save(filename, age)

# try different imputation methods
cog = np.zeros((6,2))
all_cog_cp = all_behavs_cp[:,2].astype(float)
all_cog_cp = all_cog_cp.reshape(-1)
all_cog_ic = all_behavs_ic[:,2].astype(float)
all_cog_ic = all_cog_ic.reshape(-1)
cog[0,0], cog[0,1] = cpm(all_edges_cp, all_cog_cp, seed)
cog[1,0], cog[1,1] = cpm_imp(all_edges_ic, all_cog_ic, all_edges_cp, all_cog_cp, seed, "const")
cog[2,0], cog[2,1] = cpm_imp(all_edges_ic, all_cog_ic, all_edges_cp, all_cog_cp, seed, "mean")
cog[3,0], cog[3,1] = cpm_imp_coms(all_mats_ic, all_cog_ic, all_edges_cp, all_cog_cp, seed)
cog[4,0], cog[4,1] = cpm_imp(all_edges_ic, all_cog_ic, all_edges_cp, all_cog_cp, seed)
cog[5,0], cog[5,1] = cpm_mc(all_edges_ic, all_cog_ic, all_edges_cp, all_cog_cp, seed)

filename = "../results/cnp_cog_" + str(seed) + ".npy"
np.save(filename, cog)




