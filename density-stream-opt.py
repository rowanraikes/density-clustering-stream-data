import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl
from sklearn.cluster import DBSCAN

from mpl_toolkits.mplot3d import Axes3D

class cluster:

	def __init__(self, seed_point):

		self.data_points = np.array(seed_point)
		
		self.center = seed_point
		self.weight = 1 # number of points in cluster
		self.radius = 0

	def calculate_center(self):

		self.center = np.mean(self.data_points, axis=0)


	def calculate_radius(self):

		self.radius = np.sum(np.linalg.norm(self.data_points - self.center)) / self.weight

	def get_center(self):

		return self.center

	def get_radius(self):

		return self.radius

	def get_weight(self):

		return self.weight


	def add_point(self, new_point):

		self.data_points = np.vstack((self.data_points,new_point))

		self.weight += 1

		self.calculate_center()
		
		self.calculate_radius()

	def get_points(self):

		return self.data_points


class den_stream:

	def __init__(self, eps, beta, mu):

		self.C_p = []
		self.C_o = []
		self.beta = beta
		self.eps = eps
		self.mu = mu
		self.time_elapsed = 0

	def prune_p_clusters(self):

		C_p = self.C_p

		for j in range(len(C_p)):

			if C_p[j].get_weight() < self.beta*self.mu:

				C_p.remove(C_p[j])

	def prune_o_clusters(self):

		C_o = self.C_o

		for k in range(len(C_o)):

			xi = C_o[k].get_xi(self.time_elapsed, self.T_p)

			if C_o[k].get_weight() < xi:

				C_o.remove(C_o[k])

	def update(self,p):

		self.merging(p)

		# cluster maintenance


	def euclidean(self, p1, p2):

		d = 0

		for i in range(len(p1)):

			d += (p1[i] - p2[i])**2

		return np.sqrt(d)

	def sort_distances(self, p, C):

		""" take input point and cluster array return list of indices in sorted order """

		distances = []

		for i in range(len(C)):

			distances.append(self.euclidean(p, C[i].get_center()))

		indices = np.argsort(distances)

		return indices, distances


	def merging(self,p):

		""" function takes inputs new point, updates clusters """

		# try merge p into nearest micro cluster
		C_p = self.C_p
		C_o = self.C_o

		if len(C_p) != 0:
			indices, distances = self.sort_distances(p, C_p)

			# trial closest cluster with new point
			# if new radius c_p[i] < eps -> merge p into c_p[i]
			
			curr_cluster = C_p[indices[0]]
			r_t = curr_cluster.get_radius() + self.euclidean(p, curr_cluster.get_center()) / curr_cluster.get_weight()

			if r_t <= eps:

				curr_cluster.add_point(p)

				return

		# try merge p into nearest outlier cluster	
		if len(C_o) != 0:

			indices, distances = self.sort_distances(p, C_o)

			curr_cluster = C_o[indices[0]]

			r_t = curr_cluster.get_radius() + self.euclidean(p, curr_cluster.get_center()) / curr_cluster.get_weight()

			if r_t <= eps:

				curr_cluster.add_point(p)

				# check if cluster needs to be promoted
				if curr_cluster.get_weight() > self.beta:

					C_p.append(curr_cluster)

					C_o.remove(curr_cluster)

				return 

		# no merge possible, create new outlier cluster	
		C_o.append(cluster(p))

	def de_bugging(self):

		print(str(len(self.C_p))+' core-micro clusters.' )
		print(str(len(self.C_o))+' outlier-micro clusters.' )

	
	def get_cluster_result(self):

		""" Use DBSCAN to return final clustering result 
		
		returns
		--------
		data_points : all data points making up the final cluster result

		labels : 

		"""

		# get core micro cluster centroids
		core_centroids = [cluster.get_center() for cluster in self.C_p]

		cluster_result = DBSCAN(self.eps, 1).fit(core_centroids)

		cluster_labels = cluster_result.labels_

		clustered_points = np.concatenate([cluster.get_points() for cluster in self.C_p])

		labels = []

		for i in range(len(cluster_labels)):

			labels.append(np.zeros(self.C_p[i].get_weight()) + cluster_labels[i])

		labels = np.concatenate(labels)

		print(labels)

		print(clustered_points)

		self.de_bugging()

		return labels, clustered_points

		

	def plot_lidar_frame(self, data):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.dist = 5

		ax.scatter(data[:,0], data[:,1], data[:,2], s=1)

		plt.show()

	def plot_cluster_result(self):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.dist = 5

		labels, cluster_points = self.get_cluster_result()

		ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], s=1, c=labels)
		plt.show()

# main
index = 0 # frame to load
radius = 20 # how much of frame to return
pos = [0,0,0] # location of vehicle

for index in range(1):

	data, pos = load_pkl(index, radius, pos)

# den-stream parameters 
eps = 0.1
beta = 5
mu = 4


stream = den_stream(eps, beta, mu) # instantiate stream object

stream.plot_lidar_frame(data)

# stream data points
for i in range(len(data)):

	stream.update(data[i])

stream.plot_cluster_result()



