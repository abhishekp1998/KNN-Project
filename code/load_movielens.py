import csv
import numpy as np
import os

def load_movielens_data(data_folder_path):
    """
    The MovieLens dataset is contained at data/ml-100k.zip. This function reads the
    unzipped content of the MovieLens dataset into a numpy array. The file to read in
    is called ```data/ml-100k/u.data``` The description of this dataset is:

    u.data -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	          user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC

    Return a numpy array that has size 943x1682, with each item in the matrix (i, j)
    containing the rating user i had for item j. If a user i has no rating for item j,
    you should put 0 for that entry in the matrix.

    Args:
        data_folder_path {str}: Path to MovieLens dataset (given at data/ml-100).
    Returns:
        data {np.ndarray}: Numpy array of size 943x1682, with each item in the array
            containing the rating user i had for item j. If user i did not rate item j,
            the element (i, j) should be 0.
    """
    # This is the path to the file you need to load.

    answer = np.zeros((943,1682))
    data_file = os.path.join(data_folder_path, 'u.data')
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            answer[int(row[0]) - 1][int(row[1]) - 1] = row[3]
    
    return answer
        
    
