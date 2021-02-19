import pickle, json
import numpy as np

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

	pos = poses[index]['position']

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
	object_locs = cuboid_data[['position.x', 'position.y', 'position.z']].to_numpy()


	pkl_file = open(path, 'rb')

	data = pickle.load(pkl_file).to_numpy()

	spin_indices = np.where(data[:,5] == 0)
  
	xyz_data = data[spin_indices, 0:3]


	# ground-plane extraction
	xyz_data = extract_ground_plane(xyz_data, pos['z'])

	return xyz_data, pos, object_locs

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

	return data[indices]

def extract_ground_plane(data, ground_height):

	""" remove ground plane """

	# naive approach
	indices = np.where(data[:,2] > ground_height)

	return data[indices]





