import numpy as np 
from .distances import euclidean_distances, manhattan_distances, cosine_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None 



    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        HINT: One use case of KNN is for imputation, where the features and the targets 
        are the same. See tests/test_collaborative_filtering for an example of this.
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features = features
        self.targets = targets         

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, label has shape (1, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_features)
        """
        Knearest = self.n_neighbors
        labels = None
        neighbors = None 
        targets = self.targets
        agg = self.aggregator
        NN = None

        if(ignore_first):
            Knearest += 1
        
        if (self.distance_measure == 'cosine'):
            neighbors = cosine_distances(features, self.features)

        elif (self.distance_measure == 'manhattan'):
            neighbors = manhattan_distances(features, self.features)

        else :
            neighbors = euclidean_distances(features,self.features)
        
        neighbors = np.argsort(neighbors, axis = 1)
        
        megalist = []

        for row in neighbors:
            innerlist = []
            for i in range(Knearest):
                innerlist.append(row[i])
            megalist.append(innerlist)

        NN = np.asarray(megalist)
        newTargets = None 

        outerlist = []
        for row in NN:
            innerlist = []
            for i in row: 
                innerlist.append(targets[i])
            outerlist.append(innerlist)
        
        newTargets = np.asarray(outerlist)

        outerlist = []

        for row in newTargets:
            if(agg == 'median'):
                outerlist.append(self.median(row))

            elif(agg == 'mean'):
                outerlist.append(self.mean(row))

            else:
                outerlist.append(self.mode(row))
        
        labels = np.asarray(outerlist)
        return labels


    def mode(self,targetList):
        trans = np.transpose(targetList)
        templist = []
        for row in trans:
            unique = np.unique(row, return_counts = 'true')
            most = np.argmax(unique[1])
            most = unique[0][most]
            templist.append(most)
        return np.asarray(templist)
        



    def mean(self,targetList):
        return np.mean(targetList, axis = 0)
    
    def median(self,targetList):
        return np.median(targetList,axis = 0)







        








