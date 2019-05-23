import os
import numpy as np 
dataset_path = '/home/jqz/MVDepthNet/MultiView/TUM_dataset/'
datadir = ['rgbd_dataset_freiburg1_desk/','rgbd_dataset_freiburg1_xyz/','rgbd_dataset_freiburg2_desk/','rgbd_dataset_freiburg2_desk_with_person/',
'rgbd_dataset_freiburg2_xyz/','rgbd_dataset_freiburg3_long_office_household/','rgbd_dataset_freiburg3_nostructure_texture_near_withloop/',
'rgbd_dataset_freiburg3_sitting_xyz/','rgbd_dataset_freiburg3_structure_texture_far/']
for i in range(9):
	datapath = dataset_path+datadir[i]
	data_size = len(os.listdir(datapath+'poses/'))
	for j in range(data_size-2):
		f = open('./training_list.txt','a')
		line = "{:04d}".format(i) + " " + "{:04d}".format(j) + " " + "{:04d}".format(j+1) + " " + "{:04d}".format(j+2) + "\n"
		#print line
		f.write(line)
		f.close()