import numpy as np
import copy
import time

"""Robust Matrix Completion"""

def converge(pr, dr, ep, ed):
    #print('pr = %f' % pr)
    #print('dr = %f' % dr)
    h = (pr < ep) and (dr < ed)
    return h

def shrinkage(x, kappa):
    z = np.multiply(np.sign(x),np.maximum(np.abs(x)-np.abs(kappa),0))
    return z

def RMC_l1(Z, lambda_1):
    start = time.time()
    print("admm l1")
    nrow = np.shape(Z)[0]
    ncol = np.shape(Z)[1]
    UObs = np.isnan(Z)
    Obs = ~UObs
    X = np.zeros([nrow, ncol])
    X_old = np.zeros([nrow, ncol])
    E = np.zeros([nrow, ncol])
    E_old = np.zeros([nrow, ncol])
    Y = np.zeros([nrow, ncol])
    # set the unobserved to 0, will be set back to nan after completion
    Z[UObs] = 0
    u = 0.0001
    pr = 100
    dr = 100
    ep = 0.01
    ed = 0.01
    t_inc = 2
    t_dec = 2
    mu = 10
    while ~converge(pr, dr, ep, ed):
        # update X
        T = Z-E+Y/u
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        nrs = np.shape(S)[0]
        I = np.identity(nrs)
        diag = np.maximum(np.diag(S)-1/u*I, 0)
        X_old = copy.copy(X)
        X = U@diag@Vh
        
        # update E
        E_old = copy.copy(E)
        B = Z-X+Y/u
        e = shrinkage(B, lambda_1/u)
        E[Obs] = e[Obs]
        E[UObs] = B[UObs]

        # calculate residues of this iteration
        pr = np.linalg.norm(X+E-Z)
        dr = np.linalg.norm(u*(E-E_old))

        # update Y 
        Y = Y+u*(Z-X-E)

        # update u
        if pr > mu*dr:
            u = t_inc*u
        elif dr > mu*pr:
            u = u/t_dec
        else:
            u = u

    E[UObs] = 0
    Z[UObs] = np.nan
    Z_imp = copy.copy(Z)
    Z_imp[UObs] = X[UObs]

    end = time.time()
    #print("time: %f\n" %(end-start))
    return X, Z_imp

def RMC_l2(Z, lambda_1):
    start = time.time()
    print("admm l2")
    nrow = np.shape(Z)[0]
    ncol = np.shape(Z)[1]
    UObs = np.isnan(Z)
    Obs = ~UObs
    X = np.zeros([nrow, ncol])
    X_old = np.zeros([nrow, ncol])
    E = np.zeros([nrow, ncol])
    E_old = np.zeros([nrow, ncol])
    Y = np.zeros([nrow, ncol])
    # set the unobserved to 0, will be set back to nan after completion
    Z[UObs] = 0
    u = 0.0001
    pr = 100
    dr = 100
    ep = 0.01
    ed = 0.01
    t_inc = 2
    t_dec = 2
    mu = 10
    while ~converge(pr, dr, ep, ed):
        #print('u = %f' %u)
        # update X
        T = Z-E+Y/u
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        nrs = np.shape(S)[0]
        I = np.identity(nrs)
        diag = np.maximum(np.diag(S)-1/u*I, 0)
        X_old = copy.copy(X)
        X = U@diag@Vh
  
        # update E
        E_old = copy.copy(E)
        B = Z-X+Y/u
        E[Obs] = B[Obs]*u/(2*lambda_1+u)
        E[UObs] = B[UObs]

        # calculate residues of this iteration
        pr = np.linalg.norm(X+E-Z)
        dr = np.linalg.norm(u*(E-E_old))

        # update Y and u
        Y = Y+u*(Z-X-E)
        
        # update u
        if pr > mu*dr:
            u = t_inc*u
        elif dr > mu*pr:
            u = u/t_dec
        else:
            u = u

    E[UObs] = 0
    Z[UObs] = np.nan
    Z_imp = copy.copy(Z)
    Z_imp[UObs] = X[UObs]
    
    end = time.time()
    #print("time: %f\n" % (end-start))
    return X, Z_imp
    






