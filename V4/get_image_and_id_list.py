import numpy as np
import os
from scipy.misc import imread  # (h, w, c)
from scipy.misc import imresize

def get_image_and_id_list(filepath):
	img_names = []
	labels =[]
	for img_name in os.listdir(filepath):
		if img_name.endswith('.jpg'):
			if img_name[0: 2] == '-1':
				label = int(img_name[0: 2])
			else:
				label = int(img_name[0: 4])
			img_names.append(img_name)
			labels.append(label)
		
	return img_names, labels