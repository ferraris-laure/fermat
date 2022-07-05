import numpy as np

def harmonize_points(X, method):
    """
    Given a set of d-dimensional points, it replace each point by the baricenter of the graph defined by `method`
    
    Parameters
    -----------
    X: numpy.ndarray
        2-dimension array X.shape = n, d 
    method: string ["knn", "Voronoi"]
        Options are:
                'knn'         -- ...
                'Voronoi'     --  ...
    """
    
    n, d = X.shape
    
    if d==1 and method=='Voronoi':
        X_order = np.sort(X, axis=0)
        return (X_order[1:,] + X_order[:-1,]) / 2
    else:
        raise ValueError("This method has not been implemented for these set of parameters.")