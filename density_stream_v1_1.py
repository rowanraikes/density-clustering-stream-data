import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl
from sklearn.cluster import DBSCAN
from collections import Counter
import time, pickle, json


from mpl_toolkits.mplot3d import Axes3D

"""
	density-stream v_1.1
	--------------------
	-> extends version 1.0 to LiDAR sweep angle.
	-> this means we can detect when a full sweep has been made

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
		self.consolidated_clusters = []
		self.beta = beta
		self.eps = eps
		self.time_elapsed = 0
		self.last_n = np.zeros((100,3)) # we need not store the z values
		self.ego_pos = np.zeros(3) # store vehicle location
		self.sweep_start_angle = 360
		self.prune_freq = 100
		self.done_frame = False

	def reset(self):

		self.consolidated_clusters = []
		self.done_frame = True


	def prune_clusters(self):

		active_thresh = 45
		reset_thresh = 10

		# current LiDAR orientation
		l_angle = np.degrees(self.sweep_angle(np.mean(self.last_n, axis=0)))
		
		print(self.sweep_start_angle, l_angle)

		# clustered whole sweep?
		if np.abs(self.sweep_start_angle - l_angle) < reset_thresh and np.abs(self.sweep_start_angle - l_angle) > 0:
			print('done frame!')
			self.reset()

		# sweep angle of all clusters
		c_p_angles = [np.degrees(self.sweep_angle(cluster.get_center())) for cluster in self.C_p]
		c_o_angles = [np.degrees(self.sweep_angle(cluster.get_center())) for cluster in self.C_o]

		# remove unchanging outlier clusters from memory
		delta = np.abs(c_o_angles - l_angle)
		not_remove = np.where(delta < active_thresh)[0]

		self.C_o = [self.C_o[index] for index in not_remove]

		delta = np.abs(c_p_angles - l_angle)
		not_remove = np.where(delta < active_thresh)[0]

		for cluster in not_remove:

			# append unchanging core clusters to list
			self.consolidated_clusters.append(self.C_p[cluster])

		# remove unchanging core clusters from list
		self.C_p = [self.C_p[index] for index in not_remove]

	def update(self,p,i):

		# on reset
		if i == 0:
			self.done_frame = False

		# merge p
		self.merging(p)

		# update 
		self.last_n[:-1,:] = self.last_n[1:,:]; self.last_n[-1,:] = p[:]

		# setup angle
		if i == self.prune_freq:
			self.sweep_start_angle = np.degrees(self.sweep_angle(np.mean(self.last_n, axis=0)))

		# cluster maintenance	
		if i%self.prune_freq == 0:
			self.prune_clusters()

		# return if we have clustered the whole frame
		return self.done_frame

			
	def sweep_angle(self, pos):

		# account for vehicle location
		pos = pos - self.ego_pos

		angle = np.arctan(pos[1]/pos[0])

		if pos[0] < 0:

			return (np.pi + angle)

		elif angle < 0:

			return (2*np.pi + angle)

		else:
			return angle

		

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
		core_centroids = [cluster.get_center() for cluster in self.consolidated_clusters]

		# DBSCAN for final cluster result
		cluster_result = DBSCAN(3, 1).fit(core_centroids)  
		cluster_labels = cluster_result.labels_
		cluster_indices = list(Counter(cluster_labels).keys())

		labels = []
		clustered_points = []
		bounding_boxes = []

		for index in cluster_indices:

			constituent_indices = np.where(cluster_labels==index)[0] # indices of all micro-clusters

			constituent_clusters = [self.consolidated_clusters[i] for i in constituent_indices]

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


	def plot_cluster_result(self, pos):

		# instantiate figure		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		fig.set_size_inches(18.5, 10.5)
		ax.dist = 3
		ax.view_init(elev=60, azim=45)
		ax.set_xlim3d(pos['x']-50, pos['x']+50)
		ax.set_ylim3d(pos['y']-50, pos['y']+50)
		ax.set_zlim3d(pos['z'], pos['z']+50)

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

# pkl_file = open('data/00_cuboids.pkl', 'rb')
# cuboid_data = pickle.load(pkl_file)
# object_locs = cuboid_data[['position.x', 'position.y', 'position.z']].to_numpy()
# object_sizes = cuboid_data[['dimensions.x', 'dimensions.y', 'dimensions.z']]

frame_no = 0

stream = den_stream(eps, beta) # instantiate stream object

data, ego_pos = load_pkl(frame_no, radius)

stream.ego_pos = np.array([ego_pos['x'], ego_pos['y'], ego_pos['z']])

start_time = time.time()

# stream data points
for i in range(len(data)):

	if stream.update(data[i], i):
		stream.plot_cluster_result(ego_pos)
		plt.show()
		break


	









