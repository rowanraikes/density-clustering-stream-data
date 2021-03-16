import numpy as np
import dbscan
from collections import Counter

""" 
	density-stream v_1.3
	--------------------
	-> now store only data point labels rather than points themselves in cluster object.

	-> version adapts den-stream algorithm, optimising it for use with LiDAR data.
	-> can only process one frame at a time.
	-> no cluster pruning implementation.

"""

class cluster:

	def __init__(self, seed_point, seed_label):

		self.points = np.array(seed_label)
		
		self.center = seed_point
		self.weight = 1 # number of points in cluster
		self.radius = 0

	def calculate_center(self, new_point):

		self.center = self.center*((self.weight-1)/self.weight) + new_point/self.weight


	def get_center(self):

		return self.center

	def get_weight(self):

		return self.weight


	def add_point(self, new_point, new_label):

		self.points = np.append(self.points, new_label)

		self.weight += 1

		self.calculate_center(new_point)

	def get_points(self):

		return self.points


class den_stream:

	def __init__(self, eps, beta):

		self.C_p = []
		self.C_o = []
		self.beta = beta
		self.eps = eps
		self.points = 0

	def fit_data(self,data):

		for i in range(len(data)):

			self.merging(data[i], i)

		self.points = i+1 # save how many points we have processed

	def sort_distances(self, p, C):

		""" take input point and cluster array return list of indices in sorted order """

		distances = np.linalg.norm(p - np.array([cluster.get_center() for cluster in C]), axis=1)

		indices = np.argsort(distances)

		return indices, distances


	def merging(self, p , p_label):

		""" function takes inputs new point, updates clusters """

		# try merge p into nearest micro cluster
		if len(self.C_p) != 0:

			# find closest core-micro cluster
			indices, distances = self.sort_distances(p, self.C_p)

			# trial closest cluster with new point
			# if new radius c_p[i] < eps -> merge p into c_p[i]
			
			curr_cluster = self.C_p[indices[0]]


			if np.linalg.norm(p - curr_cluster.get_center()) <= self.eps:

				curr_cluster.add_point(p, p_label)

				return

		# try merge p into nearest outlier cluster	
		if len(self.C_o) != 0:

			indices, distances = self.sort_distances(p, self.C_o)

			curr_cluster = self.C_o[indices[0]]


			if np.linalg.norm(p - curr_cluster.get_center()) <= self.eps:

				curr_cluster.add_point(p, p_label)

				# check if cluster needs to be promoted
				if curr_cluster.get_weight() > self.beta:

					self.C_p.append(curr_cluster)

					self.C_o.remove(curr_cluster)

				return 


		# no merge possible, create new outlier cluster	
		self.C_o.append(cluster(p, p_label))

	def de_bugging(self):

		print(str(len(self.C_p))+' core-micro clusters.' )
		print(str(len(self.C_o))+' outlier-micro clusters.' )


	def get_labels(self):

		self.de_bugging()

		labels = np.zeros(self.points) - 1

		# get core micro cluster centroids
		core_centroids = np.array([cluster.get_center() for cluster in self.C_p])

		# DBSCAN core clusters together for final cluster result
		cluster_labels = dbscan.fit_dbscan(core_centroids, 2*self.eps, 1)  

		# cluster_labels = cluster_result.labels_
		cluster_indices = list(Counter(cluster_labels).keys())

		for index in cluster_indices:

			constituent_indices = np.where(cluster_labels==index)[0]

			constituent_clusters = [self.C_p[i] for i in constituent_indices] # lump all these micro clusters into an array

			cluster_point_labels = np.concatenate([cluster.get_points() for cluster in constituent_clusters])

			labels[cluster_point_labels] = index

		return labels

