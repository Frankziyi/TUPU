import numpy as np
import os
# from skimage import io
from scipy.misc import imread  # (h, w, c)
from scipy.misc import imresize

def get_random_samples(img_names_list, labels_list, batch_size, filepath):
	indices = np.random.choice(len(img_names_list), replace=False, size=batch_size)
	images = []
	labels = []
	for i in indices:
		# Just for Inception_v3
		img_tmp = imread(filepath + img_names_list[i])
		images.append(imresize(img_tmp, (299, 299)))
		# images.append(imread(filepath + img_names_list[i]))
		labels.append(labels_list[i])
	images = np.array(images, dtype='f')
	labels = np.array(labels)

	return images, labels