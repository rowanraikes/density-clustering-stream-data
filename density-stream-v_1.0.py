import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl
from sklearn.cluster import DBSCAN
from collections import Counter
import time, pickle, json

from mpl_toolkits.mplot3d import Axes3D

""" density-stream v_1.0
	--------------------
	version adapts den-stream algorithm, optimising it for use with LiDAR data.

"""

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
		self.last_n = np.zeros((100,3)) # we need not store the z values
		self.ego_pos = np.zeros(3) # store vehicle location

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


	def sort_distances(self, p, C):

		""" take input point and cluster array return list of indices in sorted order """

		distances = np.linalg.norm(p - np.array([cluster.get_center() for cluster in C]), axis=1)

		indices = np.argsort(distances)

		return indices, distances


	def merging(self,p):

		""" function takes inputs new point, updates clusters """

		# try merge p into nearest micro cluster
		C_p = self.C_p
		C_o = self.C_o

		if len(C_p) != 0:

			# find closest core-micro cluster
			indices, distances = self.sort_distances(p, C_p)

			# trial closest cluster with new point
			# if new radius c_p[i] < eps -> merge p into c_p[i]
			
			curr_cluster = C_p[indices[0]]

			#w = curr_cluster.get_weight()
			#r_t = curr_cluster.get_radius()*(w/(w+1)) + (np.linalg.norm(p - curr_cluster.get_center()) / (w+1) )

			if np.linalg.norm(p - curr_cluster.get_center()) <= self.eps:

				curr_cluster.add_point(p)

				return

		# try merge p into nearest outlier cluster	
		if len(C_o) != 0:

			indices, distances = self.sort_distances(p, C_o)

			curr_cluster = C_o[indices[0]]

			#w = curr_cluster.get_weight()
			#r_t = curr_cluster.get_radius()*(w/(w+1)) + (np.linalg.norm(p - curr_cluster.get_center()) / (w+1) )


			if np.linalg.norm(p - curr_cluster.get_center()) <= self.eps:

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
		cluster_result = DBSCAN(3, 1).fit(core_centroids)  
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

		self.de_bugging()
		print(str(len(cluster_indices)) + ' final clusters.')

		return labels, clustered_points, bounding_boxes

		

	def plot_lidar_frame(self, data, pos):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		fig.set_size_inches(18.5, 10.5)
		ax.dist = 3
		ax.view_init(elev=60, azim=45)
		ax.set_xlim3d(pos['x']-60, pos['x']+60)
		ax.set_ylim3d(pos['y']-60, pos['y']+60)
		ax.set_zlim3d(pos['z'], pos['z']+60)

		# plot all data 
		ax.scatter(data[:,0], data[:,1], data[:,2], s=1)

		# plot object locations
		#ax.scatter(object_locs[:,0], object_locs[:,1], object_locs[:,2], c='r',s=4)


	def plot_cluster_result(self):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		fig.set_size_inches(18.5, 10.5)
		ax.dist = 3
		ax.view_init(elev=30, azim=45)

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

			# more horizontal lines
			for i in range(2):
				for j in range(2):
					ax.plot(X, Y[[j,j]], Z[[i,i]], color='r',  linewidth=0.5)

	
# main
radius = 0 # how much of frame to return

# den-stream parameters 
eps = 2
beta = 20

frame_no = 3

data, ego_pos = load_pkl(frame_no, radius)

stream = den_stream(eps, beta) # instantiate stream object

stream.ego_pos = np.array([ego_pos['x'], ego_pos['y'], ego_pos['z']])

start_time = time.time()

# stream data points
for i in range(n):

	stream.update(data[i])

stream.get_cluster_result()

times.append(time.time() - start_time)

print(str(i)+" points clustered in " +str(time.time() - start_time)+" seconds.") 

