import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
#################################################################
############## Imputation methods implementation ################
#################################################################

# the missing data is encoded as 0, it is better to encode as nan

def impute_edges(data, method, n_neighbors=5, weights="uniform"):
    # method: 1) knn: impute missing values using k nearest neighbor (across subjects)
    #         2) mean: impute missing values using mean values
    #         3) const: impute missing values using a constant value 

    # data should be a 2D matrix
    if (method == "mean"):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        return imp.fit_transform(data)

    # data should be a 2D matrix
    if (method == "const"):
        const = np.nanmean(data)
        imp = SimpleImputer(
            missing_values=np.nan, strategy='constant', fill_value=const)
        return imp.fit_transform(data)
    
    # data should be a 2D matrix
    if (method == "knn"):
        imp = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        return imp.fit_transform(data)
 

def impute_coms(data, m_miss, method):
    # method: 1) mean_task: replace missing connectomes using task average
    #         2) mean_sub: replace missing connectomes using subject average
      
    # data should be a 4D matrix [m,m,subject,tasks]
    if (method == "mean_sub"):
        m_miss = (m_miss==1)
        nsub = np.shape(data)[2]
        ntasks = np.shape(data)[3]
        mean_com = np.nanmean(data, axis=2)
        for i in range(nsub):
            for j in range(ntasks):
                if m_miss[i,j]: 
                    data[:,:,i,j]=mean_com[:,:,j]
        return data, mean_com
    
    if (method == "mean_task"):
        m_miss = (m_miss==1)
        nsub = np.shape(data)[2]
        ntasks = np.shape(data)[3]
        mean_com = np.nanmean(data, axis=3)
        for i in range(nsub):
            for j in range(ntasks):
                if m_miss[i,j]: 
                    data[:,:,i,j]=mean_com[:,:,i]
        return data

    
