import pickle, numpy

def load_pkl(path): 

	"""  Open pkl file of lidar data as numpy array containing xyz coordinates """

	pkl_file = open(path, 'rb')

	data = pickle.load(pkl_file)
  
	xyz_data = data.to_numpy()[:, 0:3]

	return xyz_data



