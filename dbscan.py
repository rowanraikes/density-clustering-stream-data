from sklearn.neighbors import NearestNeighbors
import numpy as np
from unpickle import load_pkl
from queue import deque

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nearest_neighbors(X, eps):

	neighborhoods = []

	for i in range(len(X)):

		d = np.linalg.norm((X[i] - X), axis=1) # compute pointwise distances

		neighbors = np.where(d <= eps)[0]

		neighborhoods.append(neighbors)

	return neighborhoods

def fit_dbscan(X, eps, min_pts):

	
	neighborhoods = nearest_neighbors(X, eps)


	# find number of neighbors
	is_core = [(len(neighbors) > min_pts) for neighbors in neighborhoods]


	labels = np.zeros(len(X)) - 1 # all samples are noise initially

	labels = dbscan_inner(is_core, neighborhoods, labels) # link core points together

	return labels


def dbscan_inner(is_core, neighborhoods, labels):

	neighb = []
	stack = [] # stor current cluster
	label_num = 0

    # loop over all labels
	for i in range(labels.shape[0]):

		if labels[i] != -1 or not is_core[i]: # skip point if it has already been assigned or isn't core point
			continue

		while True:
			

			labels[i] = label_num

			if is_core[i]:

				neighb = neighborhoods[i] 

            	# loop over neighborhood
				for i in range(neighb.shape[0]):
					v = neighb[i]
					if labels[v] == -1:
						labels[v] = label_num # assign 
						stack.append(v) # add point to stack

			if len(stack) == 0:
				break

			i = stack[-1] # search starting from last in stack
			del stack[-1] # remove point from stack

		#print('cluster ',label_num, ' done.')


		label_num += 1

	return labels

