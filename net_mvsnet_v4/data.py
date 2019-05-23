import os
import sys
import numpy as np
import imageio
import tensorflow as tf
import json
import time
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0,  1, 0 ],
                    [-math.sin(theta[1]),   0, math.cos(theta[1])]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                   
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R
def add_noise():
    noise = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
    noise_ang = (np.array((np.random.uniform(2, 6, (3,)),np.random.uniform(-6, -2, (3,)))).reshape(6,)) / 180 * np.pi
    noise_t = np.array((np.random.uniform(0.003, 0.013, (3,)),np.random.uniform(-0.013, -0.003, (3,)))).reshape(6,)
    np.random.shuffle(noise_ang)
    np.random.shuffle(noise_t)

    R_noise = eulerAnglesToRotationMatrix(noise_ang[:3])
    T_noise = np.eye(4)
    T_noise[:3,:3] = R_noise
    T_noise[:3,3] = noise_t[:3]

    return T_noise

def getImage(lines1, lines2):

	dataset_path = '/home/jqz/MVDepthNet/MultiView/TUM_dataset/'
	datadir = ['rgbd_dataset_freiburg1_desk/','rgbd_dataset_freiburg1_xyz/','rgbd_dataset_freiburg2_desk/','rgbd_dataset_freiburg2_desk_with_person/',
	'rgbd_dataset_freiburg2_xyz/','rgbd_dataset_freiburg3_long_office_household/','rgbd_dataset_freiburg3_nostructure_texture_near_withloop/',
	'rgbd_dataset_freiburg3_sitting_xyz/','rgbd_dataset_freiburg3_structure_texture_far/']
	dataset_path1 = dataset_path+datadir[int(lines1[0])]
	dataset_path2 = dataset_path+datadir[int(lines2[0])]

	lines1_1 = int(lines1[1])
	lines1_2 = int(lines1[2])
	lines1_3 = int(lines1[3])

	lines2_1 = int(lines2[1])
	lines2_2 = int(lines2[2])
	lines2_3 = int(lines2[3])

	rgb0 = None
	pose0 = None
	intrinsic0 = None
	for i in range(3):
		if i== 0:
			rgb0 = imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(lines1_1)))
			pose0, intrinsic0 = getPoseIntrinsic(dataset_path1, lines1_1)

		elif i==1:
			rgb0 = np.stack((rgb0, imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(lines1_2)))), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path1, lines1_2)
			pose0 = np.stack((pose0, poseTemp))
			intrinsic0 = np.stack((intrinsic0, intrinsicTemp))

		else:
			rgb0 = np.concatenate((rgb0, imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(lines1_3)))[np.newaxis,:]), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path1, lines1_3)
			pose0 = np.concatenate((pose0, poseTemp[np.newaxis,:]))
			intrinsic0 = np.concatenate((intrinsic0, intrinsicTemp[np.newaxis,:]))

	depth0 = imageio.imread(os.path.join(dataset_path1, "depths", "{:04d}.exr".format(lines1_1)))
	depth0 = depth0[..., np.newaxis]

	rgb1 = None
	pose1 = None
	intrinsic1 = None
	for i in range(3):
		if i== 0:
			rgb1 = imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(lines2_1)))
			pose1, intrinsic1 = getPoseIntrinsic(dataset_path2, lines2_1)
		elif i==1:
			rgb1 = np.stack((rgb1, imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(lines2_2)))), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path2, lines2_2)
			pose1 = np.stack((pose1, poseTemp))
			intrinsic1 = np.stack((intrinsic1, intrinsicTemp))
		else:
			rgb1 = np.concatenate((rgb1, imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(lines2_3)))[np.newaxis,:]), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path2, lines2_3)
			pose1 = np.concatenate((pose1, poseTemp[np.newaxis,:]))
			intrinsic1 = np.concatenate((intrinsic1, intrinsicTemp[np.newaxis,:]))			

	depth1 = imageio.imread(os.path.join(dataset_path2, "depths", "{:04d}.exr".format(lines2_1)))
	depth1 = depth1[...,np.newaxis]

	rgb = np.stack((rgb0, rgb1))
	depth = np.stack((depth0, depth1))
	noise = np.zeros((2,3,4,4))
	noise[0,0,:,:] = np.eye(4)
	noise[1,0,:,:] = np.eye(4)
	noise[0,1,:,:] = add_noise()
	noise[0,2,:,:] = add_noise()
	noise[1,1,:,:] = add_noise()
	noise[1,2,:,:] = add_noise()
	pose = np.matmul(noise,np.stack((pose0, pose1)))
	intrinsic = np.stack((intrinsic0, intrinsic1))
	pose_gt = np.stack((pose0, pose1))
	pose_input = np.stack((pose0, pose1))

	pose_Nei = pose_gt[:,1:].reshape((4,4,4))
	pose_Ref = np.tile(pose_gt[:,:1],[1,2,1,1]).reshape((4,4,4))
	pose_gt = np.matmul(la.inv(pose_Nei), pose_Ref).reshape((2,2,4,4))

	pose_Nei = pose[:,1:].reshape((4,4,4))
	pose_Ref = np.tile(pose[:,:1],[1,2,1,1]).reshape((4,4,4))
	pose_init = np.matmul(la.inv(pose_Nei), pose_Ref).reshape((2,2,4,4))

	return rgb[:1] / 255.0, depth[:1], pose_input[:1], intrinsic[:1], pose_gt[:1]

def getPoseIntrinsic(datasetPath, i):
	
	with open(os.path.join(datasetPath, "poses", "{:04d}.json".format(i))) as f:
		r_info = json.load(f)
		r_c_x = r_info["c_x"]
		r_c_y = r_info["c_y"]
		r_f_x = r_info["f_x"]
		r_f_y = r_info["f_y"]
		r_extrinsic = np.array(r_info["extrinsic"])

	intrinsic = np.array([r_f_x, r_f_y, r_c_x, r_c_y], dtype=np.float)
	poses = r_extrinsic

	return poses, intrinsic
def get_error(pose_star, pose_pred):
	R1, t1 = pose_star[:,:, 0:3, 0:3], pose_star[:,:, 0:3, 3]
	R2, t2 = pose_pred[:,:, 0:3, 0:3], pose_pred[:,:, 0:3, 3]

	ri = np.trace(np.matmul(np.transpose(R2, [0, 1, 3, 2]), R1))
	angle = np.arccos(np.minimum(1.0, np.maximum(-1.0, (ri-1)/2)))
	rotation_error = np.mean(np.abs(angle))

	translation_error = np.mean(np.sum((t1-t2)**2, axis=-1))

	return [rotation_error, translation_error]