import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
from fermat import Fermat 
from scipy.spatial import  distance_matrix
from scipy.stats import bootstrap
from scipy.stats import variation
from scipy.special import gamma
from gudhi.point_cloud.knn import KNearestNeighbors
from sklearn.datasets import make_moons, make_circles
from fermat.preprocessing import *



def fermat_simu_cylinder(n, d, alphas, x_max=1, n_simulations=1, harmonizer=None, n_harmonizers=1, k=None, eps=None, seed=None):
    
    """
    
    Compute the sample Fermat distance from 0 to x_max for different alpha values on a uniform sample in a d-dimendionnal cylinder of radius 1 and axis [0,x_max].
    
    Parameters
    -----------
    
    n : (int) sample size
    d : (int) dimension
    alphas : (numpy.array) values for the sample Fermat distance parameter
    x_max : (float) length of the cylinder axe
    n_simulations : (int) number of experiences (n_simulations samples will be simulated)
    harmonizer : (str) ['Knn', 'Delaunay', 'Epsilon_radius'] if not None, method for preprocessing the data (see harmonize_points function's help)
    n_harmonizers : (int) number of harmonizations of the data
    k : (int) number of neighbors if harmonizer=='Knn'
    eps : (float) radius of theneighbors ball if harmonizer=='Epsilon_radius'
    seed : (int) if not None, fix the seed 

    Returns
    -----------
    
    A data frame. Number of rows : (n_simulations x alphas.shape[0]) of estimated Fermat distances from 0 to x_max with columns : 'd', 'n', 'alphas', 'n_simulations', 'fermat_distance'.
    """
    
    
    if seed is not None:
        np.random.seed(seed)
        
    # set initial and final point
    x = np.zeros(d)
    y = np.zeros(d)
    y[0] = x_max
    
    for _ in np.arange(n_simulations):
        
        #X = sample_cylinder(n,d,x_min=0,x_max=x_max) #calling the function seems slower than inserting the script here
        #data simulation
        if d == 1:
            X = uniform.rvs(0,x_max,n)[:,np.newaxis]
        else:
            X = norm.rvs(0,1,(n,d-1))
            N = np.linalg.norm(X, axis=1)[:,np.newaxis]
            U = (uniform.rvs(0,1,n)**(1/(d-1)))[:,np.newaxis]
            X = U*X/N
            V = uniform.rvs(0,x_max,size=n)
            X = np.c_[V,X]
        
        if harmonizer is not None :
            for h in range(n_harmonizers):
                X = harmonize_points(X, method=harmonizer,k=k,eps=eps)
    
        X = np.r_[x.reshape(-1,d),X,y.reshape(-1,d)] #shape(n+2,d)
        distances = distance_matrix(X,X)

        Fermat_distances = []
        
        for al in alphas:
            f_exact = Fermat(alpha = al, path_method='FW') 
            f_exact.fit(distances, normalize=True, d=d) 
            fermat_dist_exact = f_exact.get_distance(0,-1) 
            Fermat_distances.append(fermat_dist_exact) 
            
        #df_ = pd.DataFrame({"d": d,
         #                       "n": n,
          #                       "alpha":alphas, 
           #                      'n_simulations': 1,
            #                     'fermat': Fermat_distances})
        
        df_ = pd.DataFrame({"d": d,
                                "n": n,
                                 "alpha":alphas, 
                                 'harmonizer': harmonizer,
                                 'k': k,
                                 'eps': eps,
                                 'n_simulations': 1,
                                 'fermat': Fermat_distances})
        
        if _ == 0:
            df = df_
        else:
            df = pd.concat([df, df_], axis=0)
    
    return df


def fermat_simu_moons(n, d, alphas, x, y, noise=0.1, n_simulations=1, harmonizer=None, n_harmonizers=1, k=None, eps=None, seed=None):
    
    """
    
    1. Compute the sample Fermat distance from x to y for different alpha values on a sample generated with 'sklearn.datasets.make_moons'.

    Parameters
    -----------
    
    n : (int) sample size 
    d : (int) dimension for the normalisation
    alphas : (numpy.array) values for the sample Fermat distance parameter
    x, y : (np.array shape(2,)) compute the Fermat distance from x to y
    noise : (float) parameter of the 'make_moons' function
    n_simulations : (int) number of experiences (n_simulations samples will be simulated)
    harmonizer : (str) ['Knn', 'Delaunay', Epsilon_radius'] if not None, method for preprocessing the data (see harmonize_points function's help)
    n_harmonizers : (int) number of harmonizations of the data
    k : (int) number of neighbors if method=='knn'
    eps : (float) radius of the neighbors ball
    seed : (int) if not None, fix the seed 

    Returns
    -----------
    
    1 data frame :
    
    A data frame of size (n_simulations x alphas.shape[0]) of estimated Fermat distances from x to y with columns : 'd', 'n', 'alphas', 'n_simulations', 'fermat_distance'
    
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    for _ in np.arange(n_simulations):
    
        #Data simulation
        X, labels = make_moons(n_samples=n,noise=noise)
        
        
        if harmonizer is not None :
            for h in range(n_harmonizers):
                X = harmonize_points(X, method=harmonizer,k=k,eps=eps)
    
        X = np.r_[x.reshape(-1,2),X,y.reshape(-1,2)] #shape(n+2,2)
        distances = distance_matrix(X,X)

        Fermat_distances = []
        
        for al in alphas:
            f_exact = Fermat(alpha = al, path_method='FW') 
            f_exact.fit(distances, normalize=True, d=X.shape[1])       
            fermat_dist_exact = f_exact.get_distance(0,-1) 
    
            Fermat_distances.append(fermat_dist_exact) 
            
        #df_ = pd.DataFrame({"d": X.shape[1],
                       #          "n": n,
                        #         "alpha":alphas, 
                         #        'n_simulations': 1,
                          #       'fermat': Fermat_distances})
        
        df_ = pd.DataFrame({"d": d,
                                "n": n,
                                 "alpha":alphas, 
                                 'harmonizer': harmonizer,
                                 'k': k,
                                 'eps': eps,
                                 'n_simulations': 1,
                                 'fermat': Fermat_distances})
        
        if _ == 0:
            df = df_
            
        else:
            df = pd.concat([df, df_], axis=0)
            
    return df


def fermat_simu_normalblobs(n, alphas, x, y, oscillations = 15, a = 3, n_simulations=1, harmonizer=None, n_harmonizers=1, k=None, eps=None, seed=None):
    
    """
    
    1. Compute the sample Fermat distance from x to y for different alpha values on a sample composed of 4 2d-Normal distributions with same covariance matrix [[0.01,0][0,0.01]] and means [0.3,0.3][0.3,0.7][0.7,0.3][0.7,0.7]. 
    2. Compute the sample Fermat distance from h(x) to h(y) in the transformed data set h(X) where h(s,t) = (s * np.cos(oscillations * s), a * t, s * np.sin(oscillations * s)).

    Parameters
    -----------

    n : (int) sample size / 4 BEWARE n is 1/4 of the final sample size
    d : (int) dimension for the normalisation
    alphas : (numpy.array) values for the sample Fermat distance parameter
    x, y : (np.array shape(2,)) compute the Fermat distance from x to y
    oscilations : (float) speed of rotation of the roll
    a : (float) dispersion of the points along the second axis
    n_simulations : (int) number of experiences (n_simulations samples will be simulated)
    harmonizer : (str) ['Knn', 'Delaunay', 'Epsilon_radius'] if not None, method for preprocessing the data (see harmonize_points function's help)
    n_harmonizers : (int) number of harmonizations of the data
    k : (int) number of neighbors if method=='knn'
    eps : (float) radius of the neighbors ball
    seed : (int) if not None, fix the seed 

    Returns
    -----------
    
    2 data frame :
    
    1. A data frame. Number of rows : (n_simulations x alphas.shape[0]) of estimated Fermat distances from x to y with columns : 'd', 'n', 'alphas', 'n_simulations', 'fermat_distance'.
    
    2. A data frame. Number of rows : (n_simulations x alphas.shape[0]) of estimated Fermat distances from h(x) to h(t) with columns : 'd', 'n', 'alphas', 'n_simulations', 'fermat_distance'.
    
    """
    
    
    if seed is not None:
        np.random.seed(seed)
    
    #set means of the 4 blobs and covariance matrix
    mean1 = [0.3, 0.3]
    mean2 = [0.3, 0.7]
    mean3 = [0.7, 0.3]
    mean4 = [0.7, 0.7]
    cov = [[0.01, 0], [0, 0.01]]
    
    for _ in np.arange(n_simulations):

        x1 = np.random.multivariate_normal(mean1, cov, n)
        x2 = np.random.multivariate_normal(mean2, cov, n)
        x3 = np.random.multivariate_normal(mean3, cov, n)
        x4 = np.random.multivariate_normal(mean4, cov, n)
        X  = np.concatenate((x1, x2, x3, x4), axis=0)
        
        XX = np.zeros((X.shape[0], 3))
        for i in range(XX.shape[0]):
            s, t = X[i, 0], X[i, 1]
            XX[i, 0] = s * np.cos(oscillations * s)
            XX[i, 1] = a * t
            XX[i, 2] = s * np.sin(oscillations * s)
        w1 = x[0] * np.cos(oscillations * x[0])
        w2 = a * x[1]
        w3 = x[0] * np.sin(oscillations * x[0])
        w = np.array([w1,w2,w3])
        
        z1 = y[0] * np.cos(oscillations * y[0])
        z2 = a * y[1]
        z3 = y[0] * np.sin(oscillations * y[0])
        z = np.array([z1,z2,z3])
        
        if harmonizer is not None :
            for h in range(n_harmonizers):
                X = harmonize_points(X, method=harmonizer,k=k,eps=eps)
                XX = harmonize_points(XX, method=harmonizer,k=k,eps=eps)
    
        X = np.r_[x.reshape(-1,2),X,y.reshape(-1,2)] #shape(n+2,2)
        distancesX = distance_matrix(X,X)
        
        XX = np.r_[w.reshape(-1,3),XX,z.reshape(-1,3)] #shape(n+2,3)
        distancesXX = distance_matrix(XX,XX)

        Fermat_distancesX = []
        Fermat_distancesXX = []
        
        for al in alphas:
            f_exactX = Fermat(alpha = al, path_method='FW') 
            f_exactXX = Fermat(alpha = al, path_method='FW') 
            
            f_exactX.fit(distancesX, normalize=True, d=X.shape[1]) 
            f_exactXX.fit(distancesXX, normalize=True, d=XX.shape[1]) 
            
            fermat_dist_exactX = f_exactX.get_distance(0,-1) 
            fermat_dist_exactXX = f_exactXX.get_distance(0,-1) 
            
            Fermat_distancesX.append(fermat_dist_exactX) 
            Fermat_distancesXX.append(fermat_dist_exactXX) 
            
        #df_X = pd.DataFrame({"d": X.shape[1],
         #                       "n": n*4,
          #                       "alpha":alphas, 
           #                      'n_simulations': 1,
            #                     'fermat': Fermat_distancesX})
        
        df_X = pd.DataFrame({"d": X.shape[1],
                                "n": n*4,
                                 "alpha":alphas, 
                                 'harmonizer': harmonizer,
                                 'k': k,
                                 'eps': eps,
                                 'n_simulations': 1,
                                 'fermat': Fermat_distancesX})
        
        #df_XX = pd.DataFrame({"d": XX.shape[1],
         #                       "n": n*4,
          #                       "alpha":alphas, 
           #                      'n_simulations': 1,
            #                     'fermat': Fermat_distancesXX})
        
        df_XX = pd.DataFrame({"d": X.shape[1],
                                "n": n*4,
                                 "alpha":alphas, 
                                 'harmonizer': harmonizer,
                                 'k': k,
                                 'eps': eps,
                                 'n_simulations': 1,
                                 'fermat': Fermat_distancesXX})
        
        if _ == 0:
            dfX = df_X
            dfXX = df_XX
        else:
            dfX = pd.concat([dfX, df_X], axis=0)
            dfXX = pd.concat([dfXX, df_XX], axis=0)
    
    return dfX, dfXX

########################################

#Process df

def bootstrap_df(col):
    bts = bootstrap((col, ), variation, confidence_level=0.9, n_resamples=200, method='basic')
    return bts.confidence_interval[0],  bts.confidence_interval[1] 

def process_df(df_):
    #df_['fermat2'] = df_['fermat']**2
    #df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha']).sum()).reset_index()
    
    #df_combined['mean'] = df_combined.apply(lambda row: row['fermat'] / row['n_simulations'], axis=1)
    #df_combined['var']  = df_combined.apply(lambda row: row['fermat2'] / row['n_simulations'] - row['mean']**2, axis=1)
    #df_combined['std']  = df_combined['var'] ** .5
    #df_combined['std/mean'] = df_combined['std'] / df_combined['mean']
       
    df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha']).apply(lambda row: pd.Series({'mean': np.mean(row['fermat']),
                                                                                             'std': np.std(row['fermat']),
                                                                                             'median': np.median(row['fermat']),
                                                                                             'n_simulations': np.sum(row['n_simulations']),
                                                                                             'variation': np.std(row['fermat'])/np.mean(row['fermat']),
                                                                                             'variation_interval': bootstrap_df(row['fermat'])})))

    
    df_combined[['variation_lower', 'variation_upper']] = pd.DataFrame(df_combined['variation_interval'].tolist(), index=df_combined.index)
    
    df_combined = df_combined.reset_index()
    
    df_combined = df_combined.sort_values(by=['d', 'n', 'alpha'], ascending=True)
    
    #df_combined['mu'] = df_combined.apply(lambda row: row['mean'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    #df_combined['mu_median'] = df_combined.apply(lambda row: row['median'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    
    df_combined = df_combined.drop(['variation_interval'], axis=1)#.reset_index()
        
    return df_combined

def process_df_Knn(df_):
    #df_['fermat2'] = df_['fermat']**2
    #df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha']).sum()).reset_index()
    
    #df_combined['mean'] = df_combined.apply(lambda row: row['fermat'] / row['n_simulations'], axis=1)
    #df_combined['var']  = df_combined.apply(lambda row: row['fermat2'] / row['n_simulations'] - row['mean']**2, axis=1)
    #df_combined['std']  = df_combined['var'] ** .5
    #df_combined['std/mean'] = df_combined['std'] / df_combined['mean']
       
    df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha', 'harmonizer', 'k']).apply(lambda row: pd.Series({'mean': np.mean(row['fermat']),
                                                                                             'std': np.std(row['fermat']),
                                                                                             'median': np.median(row['fermat']),
                                                                                             'n_simulations': np.sum(row['n_simulations']),
                                                                                             'variation': np.std(row['fermat'])/np.mean(row['fermat']),
                                                                                             'variation_interval': bootstrap_df(row['fermat'])})))

    
    df_combined[['variation_lower', 'variation_upper']] = pd.DataFrame(df_combined['variation_interval'].tolist(), index=df_combined.index)
    
    df_combined = df_combined.reset_index()
    
    df_combined = df_combined.sort_values(by=['d', 'n', 'alpha', 'k'], ascending=True)
    
    #df_combined['mu'] = df_combined.apply(lambda row: row['mean'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    #df_combined['mu_median'] = df_combined.apply(lambda row: row['median'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    
    df_combined = df_combined.drop(['variation_interval'], axis=1)#.reset_index()
        
    return df_combined

def process_df_Eps(df_):
    #df_['fermat2'] = df_['fermat']**2
    #df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha']).sum()).reset_index()
    
    #df_combined['mean'] = df_combined.apply(lambda row: row['fermat'] / row['n_simulations'], axis=1)
    #df_combined['var']  = df_combined.apply(lambda row: row['fermat2'] / row['n_simulations'] - row['mean']**2, axis=1)
    #df_combined['std']  = df_combined['var'] ** .5
    #df_combined['std/mean'] = df_combined['std'] / df_combined['mean']
       
    df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha', 'harmonizer', 'eps']).apply(lambda row: pd.Series({'mean': np.mean(row['fermat']),
                                                                                             'std': np.std(row['fermat']),
                                                                                             'median': np.median(row['fermat']),
                                                                                             'n_simulations': np.sum(row['n_simulations']),
                                                                                             'variation': np.std(row['fermat'])/np.mean(row['fermat']),
                                                                                             'variation_interval': bootstrap_df(row['fermat'])})))

    
    df_combined[['variation_lower', 'variation_upper']] = pd.DataFrame(df_combined['variation_interval'].tolist(), index=df_combined.index)
    
    df_combined = df_combined.reset_index()
    
    df_combined = df_combined.sort_values(by=['d', 'n', 'alpha', 'eps'], ascending=True)
    
    #df_combined['mu'] = df_combined.apply(lambda row: row['mean'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    #df_combined['mu_median'] = df_combined.apply(lambda row: row['median'] * row['n']**((row['alpha']-1)/row['d']), axis=1) # remove 5. from here...
    
    df_combined = df_combined.drop(['variation_interval'], axis=1)#.reset_index()
        
    return df_combined



#def process_df(df_, x_max=1):
       
#    df_combined = pd.DataFrame(df_.groupby(['d', 'n', 'alpha']).apply(lambda row: pd.Series({'mean': #np.mean(row['fermat']), 
#                                                                                             'std': #np.std(row['fermat']),
#                                                                                             'median': #np.median(row['fermat']),
                                                                                             #'n_simulations': np.sum(row['n_simulations']),
#                                                                                             'variation': #np.std(row['fermat'])/np.mean(row['fermat']),
                                                                                             #'variation_interval': bootstrap_df(row['fermat'])})))

    
#    df_combined[['variation_lower', 'variation_upper']] = #pd.DataFrame(df_combined['variation_interval'].tolist(), index=df_combined.index)
    
#    df_combined = df_combined.reset_index()
    
#    df_combined = df_combined.sort_values(by=['d', 'n', 'alpha'], ascending=True)

#for the cylinder Maybe add condition iff for other distributions
#    df_combined['mu'] = df_combined.apply(lambda row: row['mean'] / ((x_max * (np.pi**((row['d']-1)/2) / #gamma((row['d']-1)/2 + 1)))**((row['alpha']-1)/row['d'])*x_max), axis=1) # added normalization by constante of f ^beta and x_max
#    df_combined['mu_median'] = df_combined.apply(lambda row: row['median']/((x_max * (np.pi**#((row['d']-1)/2) / gamma((row['d']-1)/2 + 1)))**((row['alpha']-1)/row['d'])*x_max), axis=1) # 
##   
    
#    df_combined = df_combined.drop(['variation_interval'], axis=1)#.reset_index()
        
#    return df_combined

########################################

#Data simulation

def sample_cylinder(n,d,x_min=0,x_max=1):
    
    """
    Requires : scipy.stats.uniform and scipy.stats.norm
    
    Parameters 
    -----------
    
    n : (int) size of the sample
    d : (int) dimension of the ambiant space
    x_min, x_max : (float) lower and upper bound of the cylinder's axis
    -----------
    
    if d ==1 : Simulation of a uniform n-sample over the segment [x_min,x_max] 
    if d > 1 : Simulation of a cylinder with radius 1 and centered in the first axis from x_min to x_max 
    
    Returns
    -----------
    numpy.array of shape(n,d)
    """
    
    if d == 1:
        return uniform.rvs(x_min,x_max,n)[:,np.newaxis]
   
    X = norm.rvs(0,1,(n,d-1))
    N = np.linalg.norm(X, axis=1)[:,np.newaxis]
    U = (uniform.rvs(0,1,n)**(1/(d-1)))[:,np.newaxis]
    X = U*X/N
    V = uniform.rvs(x_min,x_max,size=n)
    X = np.c_[V,X]
    
    return X


def generate_swiss_roll(oscillations, a, n):
    """
    BEWARE MY LORD : returns the 2d version of the swiss roll (before transformation), the swiss roll and labels
    
    returns
    ---------
    xx, X, labels
    
    """
    
    mean1 = [0.3, 0.3]
    mean2 = [0.3, 0.7]
    mean3 = [0.7, 0.3]
    mean4 = [0.7, 0.7]
    cov = [[0.01, 0], [0, 0.01]]

    x1 = np.random.multivariate_normal(mean1, cov, n)
    x2 = np.random.multivariate_normal(mean2, cov, n)
    x3 = np.random.multivariate_normal(mean3, cov, n)
    x4 = np.random.multivariate_normal(mean4, cov, n)
    xx = np.concatenate((x1, x2, x3, x4), axis=0)

    labels = [0] * n + [1] * n + [2] * n + [3] * n

    X = np.zeros((xx.shape[0], 3))
    for i in range(X.shape[0]):
        x, y = xx[i, 0], xx[i, 1]
        X[i, 0] = x * np.cos(oscillations * x)
        X[i, 1] = a * y
        X[i, 2] = x * np.sin(oscillations * x)

    return xx, X, labels