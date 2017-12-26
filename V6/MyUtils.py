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


def get_random_samples(img_names_list, labels_list, batch_size, filepath, img_height, img_width):
	indices = np.random.choice(len(img_names_list), replace=False, size=batch_size)
	images = []
	labels = []
	for i in indices:
		# Just for Inception_v3
		img_tmp = imread(filepath + img_names_list[i])
		images.append(imresize(img_tmp, (img_height, img_width)))
		# images.append(imread(filepath + img_names_list[i]))
		labels.append(labels_list[i])
	images = np.array(images, dtype='f')
	labels = np.array(labels)

	return images, labels