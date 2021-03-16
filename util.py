import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unpickle import load_pkl
from sklearn.cluster import DBSCAN
from collections import Counter
import time, pickle, json
import sys
from mpl_toolkits.mplot3d import Axes3D
import itertools
from operator import itemgetter
from scipy import stats

def load_pkl(index, radius, gpe=True): 

	"""  Open pkl file of lidar data as numpy array containing xyz coordinates 
	args
	-------
	radius : radius from mean of points to return, 0 means no slice

	path : path to lidar data

	pos : ego vehicle location

	return
	-------
	xyz_data : raw uncompressed lidar data

	pos : updated ego vehicle location

	"""

	# get gps data -- need to use poses instead, will be much better
	with open('data/poses.json') as json_file:
		poses = json.load(json_file)

	ego_pos = poses[index]['position']
	pos = [ego_pos['x'], ego_pos['y'], ego_pos['z']]
	#pos = [-7,0,0]

	# construct path
	if index < 10:
		path = 'data/0'+str(index)+'.pkl'
		cube_path = 'data/0'+str(index)+'_cuboids.pkl'
	else:
		path = 'data/'+str(index)+'.pkl'
		cube_path = 'data/'+str(index)+'_cuboids.pkl'

	# get ground truth data
	cube_file = open(cube_path, 'rb')
	cuboid_data = pickle.load(cube_file)


	pkl_file = open(path, 'rb')

	data = pickle.load(pkl_file).to_numpy()[:,:3]

	indices = slice_data(data,pos,radius)

	if radius == 0:
		xyz_data = data
	else:
		xyz_data = data[indices]

	# ground-plane extraction
	xyz_data = extract_ground_plane(xyz_data, pos[2])

	return xyz_data, pos, cuboid_data

def slice_data(data, pos, radius):

	""" slice data: return only data points contained by sphere of specified radius, centered on ego vehicle.

	inputs
	--------
	data : lidar data to slice

	pos : position of ego vehicle within data

	radius : radius to slice

	returns
	--------
	data : relevant data points


	"""
	
	norms = np.linalg.norm(pos-data, axis=1)

	indices = np.where(norms < radius)

	return indices

def extract_ground_plane(data, ground_height):

	""" remove ground plane """

	# naive approach
	indices = np.where(data[:,2] > ground_height)

	return data[indices]

def plot_with_boxes(labels, data, bounding_boxes, projection):

	# instantiate figure		
	fig = plt.figure()

	if projection == '3D':

		ax = fig.add_subplot(111, projection='3d')
		fig.set_size_inches(18.5, 10.5)
		ax.dist = 2
		ax.view_init(elev=40, azim=45)

		#ax.set_xlim3d(pos[0]-150, pos[0]+150)
		#ax.set_ylim3d(pos[1]-150, pos[1]+150)
		#ax.set_zlim3d(pos[2]-50, pos[2]+100)

		# plot all data 
		ax.scatter(data[:,0], data[:,1], data[:,2], s=1, c=labels)

		# plot bounding boxes
		for box in bounding_boxes:


			for i in range(8):
				for j in range(8):
					ax.plot([box[i,0], box[j,0]], [box[i,1], box[j,1]], [box[i,2], box[j,2]], color='r',  linewidth=0.5)


	elif projection == '2D':

		ax = fig.add_subplot(111)

		# plot points
		ax.scatter(data[:,0], data[:,1], s=1, c=labels)

		# plot bounding boxes
		for box in bounding_boxes:

			for i in range(8):
				for j in range(8):
					ax.plot([box[i,0], box[j,0]], [box[i,1], box[j,1]], color='r',  linewidth=0.5)


		

def validate(data, bound_boxes, labels):

	points_list = []

	precisions = []
	recalls = []

	thresh = 100 # fewer than this many points in box --> discount score

	# for each ground truth cluster
	for box in bound_boxes:

		points = [] # list of points contained by bounding box

		for i in range(len(data)):

			if contains(box, data[i]):
				points.append(i)

		# catch empty/sparse bound box error
		if len(points) < thresh:
			continue

		points_labels = labels[np.array(points)]

		cluster_label = stats.mode(points_labels).mode

		precision = len(np.where(points_labels == cluster_label)[0]) / len(np.where(labels == cluster_label)[0])

		precisions.append(precision)

		recall = len(np.where(points_labels == cluster_label)[0]) / len(points_labels)

		recalls.append(recall)

		print(cluster_label)

		#print(precision, recall)

		#print(points_labels)
		points_list.append(points)

	return np.mean(precisions), np.mean(recalls)



def contains(bound_box, point):

	""" test if point contained by 3D bounding box"""


	p1 = bound_box[0]
	p5 = bound_box[1]
	p4 = bound_box[4]
	p2 = bound_box[6]

	u = p2 - p1
	v = p4 - p1
	w = p5 - p1

	if (np.dot(u,point) < np.dot(u,p2)) and (np.dot(u,point) > np.dot(u,p1)):
		if (np.dot(v,point) < np.dot(v,p4)) and (np.dot(v,point) > np.dot(v,p1)):
			if (np.dot(w,point) < np.dot(w,p5)) and (np.dot(w,point) > np.dot(w,p1)):
				return True

	return False



def generate_bound_boxes(positions, dimensions, yaws):

	verts = np.column_stack((positions[:,0] - (dimensions[:,0] / 2),positions[:,0] + (dimensions[:,0] / 2), positions[:,1] - (dimensions[:,1] / 2),positions[:,1] + (dimensions[:,1] / 2), positions[:,2] - (dimensions[:,2] / 2),positions[:,2] + (dimensions[:,2] / 2)))
	
	bound_boxes = []
	perp_vectors = []

	for i in range(len(positions)):

		box = np.zeros((8,3))

		# verts = (x_min, x_max, y_min, ...)

		# first generate non-rotated bounding box
		box[0:4,0] = verts[i,0] # x_min
		box[4:,0] = verts[i,1] # x_max

		box[:,1] = verts[i,2] # y_min
		box[2:4,1] = verts[i,3]
		box[6:,1] = verts[i,3]

		box[1::2,2] = verts[i,5]
		box[0::2,2] = verts[i,4] # z_min

		# non rotated vectors
		vects = np.zeros((2,3))
		vects[0,1] = 1
		vects[1,0] = 1

		# rotate bounding box
		theta = -yaws[i]
		centre = positions[i]

		# tranform to origin
		box[:,0] = (box[:,0] - centre[0])
		box[:,1] = (box[:,1] - centre[1])

		# rotation matrix
		r_m = np.zeros((3,3))
		r_m[0,:] = [np.cos(theta), -np.sin(theta), 0]
		r_m[1,:] = [np.sin(theta), np.cos(theta), 0] 
		r_m[2,:] = [0,0,1]
		
		# perform rotations
		box = np.dot(box, r_m)
		vects = np.dot(vects, r_m)

		# undo transformation
		box[:,0] = (box[:,0] + centre[0])
		box[:,1] = (box[:,1] + centre[1])

		bound_boxes.append(box)
		perp_vectors.append(vects)


	return bound_boxes

def generate_labels(bound_boxes, xyz_data):

	labels = np.zeros(len(xyz_data)) - 1

	for i, box in enumerate(bound_boxes):

		for j in range(len(xyz_data)):

			if contains(box, xyz_data[j]):
				labels[j] = i

	return labels





#   def perm_validate(labels, panda_labels):

# 	""" Enumerate all swap permuations calculating alignment score for each, 
# 	    return best possible alignment score for two lists """

# 	score = 0

# 	labels_types = range(len(list(Counter(labels).keys()))-1)
# 	panda_types = list(Counter(panda_labels).keys())

# 	print(len(panda_types)-1, ' ground truth clusters.')
# 	print(len(labels_types), ' clusters found.')

# 	all_perms = spermutations(len(list(Counter(labels).keys()))-1)

# 	for perm in all_perms:

# 		# permutate types
# 		perm_types = list(perm[0])
# 		perm_labels = labels.copy()

		

# 		# apply permutation
# 		for i in range(len(perm_types)):

# 			indices = np.where(labels == labels_types[i])[0] # find all instances in labels

# 			perm_labels[indices] = perm_types[i] # replace all

# 		new_score = len(np.where((perm_labels-panda_labels) == 0)[0]) / len(labels)

# 		if new_score > score:
# 			score = new_score

# 	return score


# xyz_data, pos, cuboid_data = load_pkl(0,20)

# print(cuboid_data.columns.values)

# box_pos = cuboid_data[['position.x','position.y','position.z']].to_numpy()
# box_size = cuboid_data[['dimensions.x', 'dimensions.y', 'dimensions.z']].to_numpy()
# box_yaw = cuboid_data[['yaw']].to_numpy()

# indices = slice_data(box_pos,pos,20)

# boxes = generate_bound_boxes(box_pos[indices], box_size[indices], box_yaw[indices])

# labels = generate_labels(boxes, xyz_data)


# np.set_printoptions(threshold=sys.maxsize)
# print(labels)


# plot_lidar_frame(xyz_data, pos, boxes)



# plt.show()



