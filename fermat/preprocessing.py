import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial import Delaunay
from gudhi.point_cloud.knn import KNearestNeighbors
from sklearn.neighbors import radius_neighbors_graph



def harmonize_points(X, method, k=5, eps=0.1, include_self=False):
    """
    
    Given a set of d-dimensional points, replaces each point by the baricenter of the graph defined by `method`. Refer to the Notebook "Harmonizer.ipynb" for more details and visualisations.
    
    Parameters
    -----------
    X : (numpy.ndarray) 2-dimension array X.shape = (n, d) 
    method : (string ['Knn', 'Delaunay', 'Epsilon_radius'])
        Options are
                if d > 1
                
                'Knn'            -- compute the baricenter of the (k-1)-neighbors of each point.
                'Delaunay'       -- compute the baricenter of the Delaunay graph. Each point is replaced by the baricenter of the sample points sharing an edge with this point.
                'Epsilon_radius' -- compute for each point the baricenter of its neighbors standing at a distance less than epsilon .
                
                if d==1       -- compute for each X[i] <-- (X[i-1]+X[i+1])/2.
                
    k : (int) If method == 'Knn', k is the number of neighbors. By default k=5.
    
    eps : (float) If method == 'Epsilon_radius', eps is the radius to define the neighbors at distance less than eps from each point.
    
    include_self : (boolean) To include or not each point in the baricenter computation.
    
    Returns
    ---------
    numpy.array of shape (n,d)
    
    """
    
    n, d = X.shape
    
    if d==1:
        X_order = np.sort(X, axis=0)
        return (X_order[1:,] + X_order[:-1,]) / 2
       
    elif d>1 and method=='Knn':
        knn = KNearestNeighbors(k) #for each point returns the indexes [i,] of the k-nn
        ind = knn.fit_transform(X) #array of shape(n,k)
        #beware : the first knn is the point itself
        if include_self==False:
            ind = ind[:,1:]
        H = X[ind].mean(axis=1) #X[ind] is of shape(X.shape[0],k,d)
        return H
    
    elif d>1 and method=='Delaunay':
        tri = Delaunay(X)
        S = tri.simplices
        H = []
        for i in np.arange(n):
            ids = np.where(S==i)[0] #indices of simplices with vertex X_i
            idx = S[ids] #array of vertices indices sharing a simplex with X_i
            idx = np.unique(idx) #removing double indices
            if include_self==False:
                idx = np.setdiff1d(idx,i) #removing i
            H.append(X[idx].mean(axis=0))
        H = np.array(H)
        return H
    
    elif d>1 and method=='Epsilon_radius':
        A = radius_neighbors_graph(X, radius=eps, include_self=include_self)
        H = []
        for i in np.arange(n):
            idx = A.toarray()[i,:].reshape(-1,1)
            idx = np.where(idx>0)[0]
            if idx.shape[0]==0:
                H.append(X[i])
            else:
                H.append(X[idx].mean(axis=0))
        H = np.array(H)
        return H
             
    else:
        raise ValueError("This method has not been implemented for these set of parameters.")