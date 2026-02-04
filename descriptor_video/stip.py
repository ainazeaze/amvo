import numpy as np

def read_stip_file(file):
	keypoints = []
	descriptors = []
	with open(file) as f:
		_ = f.readline()
		for l in f:
			if len(l.strip()) > 0:
				l = l.strip().split(' ')
				keypoints.append([int(k) for k in l[4:9]])
				descriptors.append([float(d) for d in l[9:]])
	if len(descriptors) == 0:
		return None, None
	else:
		return np.array(keypoints, dtype=np.int32), np.array(descriptors, dtype=np.float32)
