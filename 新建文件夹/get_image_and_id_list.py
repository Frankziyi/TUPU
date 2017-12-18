import numpy as np
import os

def get_image_and_id_list(Setlist):
	img_names = []
	labels =[]
	for img_name in os.listdir(filepath):
		if img_name.endswith('.jpg'):
			label = int(img_name[0: 3])
			img_names.append(img_name)
			labels.append(label)
		
	return img_names, labels