import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl
from sklearn.cluster import DBSCAN
from collections import Counter

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

		self.radius = np.sum(np.linalg.norm(self.data_points - self.center, axis=1)) / self.weight

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

	def __init__(self, eps, beta):

		self.C_p = []
		self.C_o = []
		self.beta = beta
		self.eps = eps
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

	def sort_distances(self, p, C):

		""" take input point and cluster array return list of indices in sorted order """

		#distances = []

		#for i in range(len(C)):

		#	distances.append(self.euclidean(p, C[i].get_center()))

		distances = np.linalg.norm(p - np.array([cluster.get_center() for cluster in C]), axis=1)

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
			r_t = curr_cluster.get_radius() + np.linalg.norm(p - curr_cluster.get_center()) / curr_cluster.get_weight()

			if r_t <= eps:

				curr_cluster.add_point(p)

				return

		# try merge p into nearest outlier cluster	
		if len(C_o) != 0:

			indices, distances = self.sort_distances(p, C_o)

			curr_cluster = C_o[indices[0]]

			r_t = curr_cluster.get_radius() + np.linalg.norm(p - curr_cluster.get_center()) / curr_cluster.get_weight()

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

		bounding boxes : 

		"""

		# get core micro cluster centroids
		core_centroids = [cluster.get_center() for cluster in self.C_p]

		# DBSCAN for final cluster result
		cluster_result = DBSCAN(2*self.eps+0.1, 1).fit(core_centroids)  
		cluster_labels = cluster_result.labels_
		cluster_indices = list(Counter(cluster_labels).keys())

		labels = []
		clustered_points = []
		bounding_boxes = []

		for index in cluster_indices:

			constituent_indices = np.where(cluster_labels==index)[0] # indices of all micro-clusters

			constituent_clusters = [self.C_p[i] for i in constituent_indices]

			cluster_points = np.concatenate([cluster.get_points() for cluster in constituent_clusters])

			labels.append(np.zeros(len(cluster_points)) + index) # append to labels array
			clustered_points.append(cluster_points) # append to list of all points

			# bounding box vertices
			bounding_boxes.append([np.amin(cluster_points[:,0]), np.amax(cluster_points[:,0]), np.amin(cluster_points[:,1]), np.amax(cluster_points[:,1]), np.amin(cluster_points[:,2]), np.amax(cluster_points[:,2])])


		labels = np.concatenate(labels)
		clustered_points = np.concatenate(clustered_points)


		# clustered_points = np.concatenate([cluster.get_points() for cluster in self.C_p])

		# labels = []

		# for i in range(len(cluster_labels)):

		# 	labels.append(np.zeros(self.C_p[i].get_weight()) + cluster_labels[i])

		# labels = np.concatenate(labels)

		# print(str(cluster_indices[-1] + 1) +' final clusters.' )

		# print(clustered_points)

		self.de_bugging()

		return labels, clustered_points, bounding_boxes

		

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

		labels, cluster_points, bounding_boxes = self.get_cluster_result()

		# plot points
		ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], s=1, c=labels)

		# plot bounding boxes
		for cluster_box in bounding_boxes:

			X = np.array([cluster_box[0], cluster_box[1]])
			Y = np.array([cluster_box[2], cluster_box[3]])
			Z = np.array([cluster_box[4], cluster_box[5]])

			# vertical lines
			for i in range(2):
				for j in range(2):
					ax.plot(X[[i,i]], Y[[j,j]], Z, color='r', linewidth=0.5)

			# horizontal lines
			for i in range(2):
				for j in range(2):
					ax.plot(X[[i,i]], Y, Z[[j,j]], color='r',  linewidth=0.5)

			# vertical lines
			for i in range(2):
				for j in range(2):
					ax.plot(X, Y[[j,j]], Z[[i,i]], color='r',  linewidth=0.5)

		plt.show()

	def plot_micro_clusters(self):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.dist = 5

		for cluster in self.C_p:

			points = cluster.get_points()

			ax.scatter(points[:,0], points[:,1], points[:,2], s=1)

		plt.show()


# main
radius = 30 # how much of frame to return
pos = [0,0,0] # location of vehicle

for index in range(1):

	data, pos = load_pkl(index, radius, pos)

# den-stream parameters 
eps = 0.7
beta = 10

stream = den_stream(eps, beta) # instantiate stream object

# stream data points
for i in range(len(data)):

	stream.update(data[i])

#stream.plot_micro_clusters()
stream.plot_cluster_result()



