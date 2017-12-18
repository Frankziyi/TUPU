import numpy as np
import os
import tensorflow as tf
import get_label

def Partion_TrainSet_And_ValidationSet(filepath, percentage=0.9):
	img_names = []
	for img_name in os.listdir(filepath):
		if img_name.endswith('.jpg'):
			img_names.append(img_name)

	# TrainSet_Shuffled_Tensor = tf.random_shuffle(img_names, seed)
	# # TrainSet_Shuffled is a tensor, we should turn it to a list
	# TrainSet_Shuffled_List = []
	# for i in TrainSet_Shuffled_Tensor:
	# 	TrainSet_Shuffled_List.append(str(TrainSet_Shuffled_Tensor[i]))
	# flag = int(len(TrainSet_Shuffled_List) * percentage)
	# TrainSet = TrainSet_Shuffled_List[0: flag]
	# ValidationSet = TrainSet_Shuffled_List[flag + 1: ]
	TrainSetNum = int(len(img_names) * percentage)
	indices = np.random.choice(len(img_names), replace=False, size=TrainSetNum)
	TrainSet = []
	ValidationSet = []
	for i in indices:
		TrainSet.append(img_names[i])

	for i in range(len(img_names)):
		if i not in indices:
			ValidationSet.append(img_names[i])

	return TrainSet, ValidationSet