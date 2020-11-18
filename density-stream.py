import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl

from mpl_toolkits.mplot3d import Axes3D

class cluster:

	def __init__(self, seed_point):

		self.data_points = np.array(seed_point)
		self.center = seed_point
		self.weight = self.f(0)
		self.radius = 0
		self.t_o = seed_point[3]

	def f(self,t):

		lmbda = 1

		return np.power(2, -lmbda*t)

	def calculate_center(self):

		self.center = np.mean(self.data_points[:,:2], axis=0)

	def calculate_weight(self, t_curr):

		time_array = np.subtract(self.data_points[:,3], t_curr)

		self.weight = np.sum(self.f(time_array))

	def calculate_radius(self, t_curr):

		distances = []

		for i in range(len(self.data_points)):

			distances.append(euclidean(self.data_points[i,:2], self.center))

		self.radius = np.sum(np.multiply(self.f(np.subtract(self.data_points[:,3], t_curr)), distances)) / self.weight

	def get_center(self):

		return self.center

	def get_radius(self):

		return self.radius

	def get_weight(self):

		return self.weight

	def get_xi(self, t_curr, T_p, lmbda):

		return (2**(-lmbda*(t_curr - self.t_o + T_p)) - 1) / ( 2**(-lmbda*T_p) - 1)



	def add_point(self, new_point):

		t_curr = new_point[3]

		self.data_points = np.vstack((self.data_points,new_point))

		self.calculate_center()
		self.calculate_weight(t_curr)
		self.calculate_radius(t_curr)

	def get_points(self):

		return self.data_points


class den_stream:

	def __init__(self, eps, beta, mu, lmbda):

		self.C_p = []
		self.C_o = []
		self.lmbda = lmbda
		self.beta = beta
		self.eps = eps
		self.mu = mu
		self.T_p = (1/lmbda)*np.log((beta*mu)/(beta*mu-1)) # checking time period
		self.time_elapsed = 0

	def prune_p_clusters(self):

		C_p = self.C_p

		for j in range(len(C_p)):

			if C_p[j].get_weight() < self.beta*self.mu:

				C_p.remove(C_p[j])

	def prune_o_clusters(self):

		C_o = self.C_o

		for k in range(len(C_o)):

			xi = C_o[k].get_xi(self.time_elapsed, self.T_p, self.lmbda)

			if C_o.get_weight() < xi:

				C_o.remove(C_o[k])

	def update(self,p):

		self.merging(np.append(p, self.time_elapsed))

		self.time_elapsed += 1

	def euclidean(self, p1, p2):

		d = 0

		for i in range(len(p1)):

			d += (p1[i] - p2[i])**2

		return np.sqrt(d)

	def sort_distances(self, p, C):

		""" take input point and cluster array return list of indices in sorted order """

		distances = []

		for i in range(len(C)):

			distances.append(self.euclidean(p[:2], C[i].get_center()))

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
			r_t = curr_cluster.get_radius() + self.euclidean(p[:2], curr_cluster.get_center()) / curr_cluster.get_weight()

			if r_t <= eps:

				curr_cluster.add_point(p)

				return

		# try merge p into nearest outlier cluster	
		if len(C_o) != 0:

			indices, distances = self.sort_distances(p, C_o)

			curr_cluster = C_o[indices[0]]

			r_t = curr_cluster.get_radius() + self.euclidean(p[:2], curr_cluster.get_center()) / curr_cluster.get_weight()

			if r_t <= eps:

				curr_cluster.add_point(p)

				# check if cluster needs to be promoted
				if curr_cluster.get_weight() > self.beta*self.mu:

					C_p.append(curr_cluster)

					C_o.remove(curr_cluster)

				return 

		# no merge possible, create new outlier cluster	
		C_o.append(cluster(p))


	def directly_density_reachable(self, c1, c2, eps):

		""" given two clusters; c1, c2; return True if they are directly density reachable.  """

		if euclidean(c1.get_center(), c2.get_center()) < 2*eps:

			return True
		else:
			return False

	def density_connected(self, initial_cluster):

		""" given a core cluster, return all clusters reachable from it """

		curr_clusters = [initial_cluster]

		j = 0

		# loop over all points in cluster
		while j < len(curr_clusters):
	
			# loop over all unclustered points
			for i, cluster in enumerate(self.C_p):

				if cluster not in curr_clusters and self.directly_density_reachable(cluster, curr_clusters[j], eps):

					curr_clusters.append(cluster)

					self.C_p.remove(cluster)

			j += 1
			
		return curr_clusters

	def get_cluster_result(self):

		""" return the current clustering result """

		# get all core clusters
		final_clusters = []
		C_p = self.C_p

		# iterate until all clusters have been processed
		while len(C_p) != 0:

			# loop over cluster array

			seed_cluster = C_p[0]

			clusters = self.density_connected(seed_cluster)

			final_clusters.append(clusters)

			C_p.remove(seed_cluster)


		return final_clusters


	

def euclidean(p1, p2):

	d = 0

	for i in range(len(p1)):

		d += (p1[i] - p2[i])**2

	return np.sqrt(d)


# main
path = 'data/00.pkl'
data = load_pkl(path)

eps = 1
beta = 2
mu = 1
lmbda = 1

points_to_use = 20000

threshold_range = 60

stream = den_stream(eps, beta, mu, lmbda)

# loop over data
for i in range(len(data)):

	# only use points within given range and ground plane extraction
	if euclidean(data[i], [0,0,0]) < threshold_range and data[i,2] > 0.2:

		stream.update(data[i])

	if i == points_to_use:

		result = stream.get_cluster_result()

		break

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.dist = 5


# plot clusters
for i in range(len(result)):

	cluster_list = result[i]

	cluster_points = cluster_list[0].get_points()

	if len(cluster_list) > 1:		

		for j in range(1,len(cluster_list)):

			cluster_points = np.vstack((cluster_points, cluster_list[j].get_points()))

	
	ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], s=1)





plt.show()
