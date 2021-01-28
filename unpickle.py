import pickle, json
import numpy as np

def load_pkl(index, radius, pos): 

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

	# get gps data
	with open('data/gps.json') as json_file:
		gps = json.load(json_file)

	# get time stamps
	with open('data/timestamps.json') as json_file:
		timestamps = json.load(json_file)

	# construct path
	if index < 10:
		path = 'data/0'+str(index)+'.pkl'
	else:
		path = 'data'+str(index)+'.pkl'

	pkl_file = open(path, 'rb')

	data = pickle.load(pkl_file)
  
	xyz_data = data.to_numpy()[:, 0:3]

	# slice data
	if radius != 0:
		xyz_data = slice_data(xyz_data, pos, radius)

	# ground-plane extraction
	xyz_data = extract_ground_plane(xyz_data, 0)

	# update pos
	delta_t = timestamps[index+1] - timestamps[index]

	pos[0] = pos[0] + delta_t*gps[index]['xvel'] # x
	pos[1] = pos[1] + delta_t*gps[index]['yvel'] # y

	pos[2] = gps[index]['height'] # z

	return xyz_data, pos

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





