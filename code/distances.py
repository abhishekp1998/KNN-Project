import numpy as np
import copy

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
      
      
    
    megalist = [] 

    for rowx in X:
        innerlist = []
        for rowy in Y:
            distance = np.sqrt(np.sum(np.square(rowx - rowy)))
            innerlist.append(distance)
        megalist.append(copy.deepcopy(innerlist))
    return np.asarray(megalist)

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    megalist = [] 

    for rowx in X:
        innerlist = []
        for rowy in Y:
            distance = np.sum(np.abs(rowx - rowy))
            innerlist.append(distance)
        megalist.append(copy.deepcopy(innerlist))
    return np.asarray(megalist)

def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    megalist = [] 

    for rowx in X:
        innerlist = []
        for rowy in Y:
            trans = rowx@np.transpose(rowy)
            unitX = np.sqrt(np.sum(np.square(rowx)))
            unitY = np.sqrt(np.sum(np.square(rowy)))
            distance = 1 -trans/(unitX * unitY)
            innerlist.append(distance)
        megalist.append(copy.deepcopy(innerlist))
    return np.asarray(megalist)

