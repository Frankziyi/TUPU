import numpy as np
import os

def get_label(ImageNameList):
	labels =[]
	for img_name in ImageNameList:
		label = int(img_name[0: 4])
		labels.append(label)
		
	return labels