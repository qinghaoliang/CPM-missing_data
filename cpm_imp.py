import numpy as np
from corr_pval import corr_pval
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, roc_auc_score
from Imputation import impute_edges


"""CPM impute data at edge level
Input:
all_edges_ic: edges of incomplete subjects
all_behav_ic: phenotype of incomplete subjects
all_edges: edges of complete subjects
all_behav: phenotype of complete subjects
seed: random number for cross-validation data split
method: 1) knn: impute missing values using k nearest neighbor (across subjects)
        2) mean: impute missing values using mean values
        3) const: impute missing values using a constant value  
thresh: feature selection threshold, p-value
alphas_ridge: ridge regression parameters 

Output: 
Rcv for regression. Roc_auc for classifciation
"""

def cpm_imp(all_edges_ic, all_behav_ic, all_edges, all_behav, seed, method="knn",
            n_neighbors=5, weights="uniform", thresh=0.1, 
            alphas_ridge=10**np.linspace(3, -10, 50)):
    """linear/ridge regression"""

    # all_behav should be array of size (n,)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_linear = np.zeros(n_splits)
    r2_ridge = np.zeros(n_splits)

    k = 0
    for train_index, test_index in kf.split(all_edges):
        # split the data into training and test
        x_test, y_test = all_edges[test_index], all_behav[test_index]
        x_train, y_train = all_edges[train_index], all_behav[train_index]
        x_train = np.concatenate((all_edges_ic, x_train), axis=0)
        y_train = np.concatenate((all_behav_ic, y_train), axis=0)
        
        # feature selection
        edges_corr, edges_p = corr_pval(x_train, y_train)
        edges_select = edges_p < thresh 
        x_train = x_train[:, edges_select]
        x_test = x_test[:, edges_select]
        edges_corr = edges_corr[edges_select]

        # impute missing values
        x_train = impute_edges(x_train, method, n_neighbors, weights)

        # linear regression
        edges_pos = edges_corr > 0
        edges_neg = edges_corr < 0   
        train_sum = np.sum(x_train[:,edges_pos], axis=1) - np.sum(x_train[:,edges_neg], axis=1)
        train_sum = train_sum.reshape(-1,1)
        linear_reg = LinearRegression().fit(train_sum, y_train)
        test_sum = np.sum(x_test[:,edges_pos], axis=1) - np.sum(x_test[:,edges_neg], axis=1)
        test_sum = test_sum.reshape(-1,1)
        y_pred_linear = linear_reg.predict(test_sum)
        r2_linear[k] = r2_score(y_test, y_pred_linear)
 
        # ridge regression
        ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
        ridge_grid.fit(x_train, y_train)
        y_pred_ridge = ridge_grid.predict(x_test)
        r2_ridge[k] = r2_score(y_test, y_pred_ridge)

        #update the iteration counts
        k=k+1
       
    r2_linear[r2_linear<0] = 0
    r_linear = np.mean(np.sqrt(r2_linear))
    r2_ridge[r2_ridge < 0] = 0
    r_ridge = np.mean(np.sqrt(r2_ridge))

    return r_linear, r_ridge


def cpm_imp_clf(all_edges_ic, all_behav_ic, all_edges, all_behav, seed, method="knn",
            n_neighbors=5, weights="uniform", thresh=0.1):
    """svc classicification"""

    # all_behav should be array of size (n,)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    roc_svm = np.zeros(n_splits)

    k = 0
    for train_index, test_index in kf.split(all_edges):
        # split the data into training and test
        x_test, y_test = all_edges[test_index], all_behav[test_index]
        x_train, y_train = all_edges[train_index], all_behav[train_index]
        x_train = np.concatenate((all_edges_ic, x_train), axis=0)
        y_train = np.concatenate((all_behav_ic, y_train), axis=0)
        
        # feature selection
        edges_corr, edges_p = corr_pval(x_train, y_train)
        edges_select = edges_p < thresh 
        x_train = x_train[:, edges_select]
        x_test = x_test[:, edges_select]
        edges_corr = edges_corr[edges_select]

        # impute missing values
        x_train = impute_edges(x_train, method, n_neighbors, weights)

        # svc classifier
        svc_clf = LinearSVC()
        svc_clf.fit(x_train, y_train)
        #y_pred = svc_clf.predict(x_test)
        roc_svm[k] = roc_auc_score(y_test, svc_clf.decision_function(x_test))

        #update the iteration counts
        k=k+1
       
    roc_svm = np.mean(roc_svm)

    return roc_svm
