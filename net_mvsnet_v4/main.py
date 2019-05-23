import os
import sys
import numpy as np
import imageio
import tensorflow as tf
import json
import time
import matplotlib.pyplot as plt

from network import Network
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
    noise_ang = (np.array((np.random.uniform(0.3, 0.5, (3,)),np.random.uniform(-0.5, -0.3, (3,)))).reshape(6,)) / 180 * np.pi
    noise_t = np.array((np.random.uniform(0.005, 0.01, (3,)),np.random.uniform(-0.01, -0.005, (3,)))).reshape(6,)
    np.random.shuffle(noise_ang)
    np.random.shuffle(noise_t)

    R_noise = eulerAnglesToRotationMatrix(noise_ang[:3])
    T_noise = np.eye(4)
    T_noise[:3,:3] = R_noise
    T_noise[:3,3] = noise_t[:3]

    return T_noise

def getImage():

	dataset_path1 = '/home/cjy/workspace/pytorch/DeepMVS/dataset/train/rgbd_20_to_inf_3d_train/0016/'
	dataset_path2 = '/home/cjy/workspace/pytorch/DeepMVS/dataset/train/rgbd_20_to_inf_3d_train/0139/'

	rgb0 = None
	pose0 = None
	intrinsic0 = None
	for i in range(3):
		if i== 0:
			rgb0 = imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(i)))
			pose0, intrinsic0 = getPoseIntrinsic(dataset_path1, i)

		elif i==1:
			rgb0 = np.stack((rgb0, imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(i)))), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path1, i)
			pose0 = np.stack((pose0, poseTemp))
			intrinsic0 = np.stack((intrinsic0, intrinsicTemp))

		else:
			rgb0 = np.concatenate((rgb0, imageio.imread(os.path.join(dataset_path1, "images", "{:04d}.png".format(i)))[np.newaxis,:]), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path1, i)
			pose0 = np.concatenate((pose0, poseTemp[np.newaxis,:]))
			intrinsic0 = np.concatenate((intrinsic0, intrinsicTemp[np.newaxis,:]))

	depth0 = imageio.imread(os.path.join(dataset_path1, "depths", "{:04d}.exr".format(0)))
	depth0 = depth0[..., np.newaxis]

	rgb1 = None
	pose1 = None
	intrinsic1 = None
	for i in range(3):
		if i== 0:
			rgb1 = imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(i)))
			pose1, intrinsic1 = getPoseIntrinsic(dataset_path2, i)
		elif i==1:
			rgb1 = np.stack((rgb1, imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(i)))), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path2, i)
			pose1 = np.stack((pose1, poseTemp))
			intrinsic1 = np.stack((intrinsic1, intrinsicTemp))
		else:
			rgb1 = np.concatenate((rgb1, imageio.imread(os.path.join(dataset_path2, "images", "{:04d}.png".format(i)))[np.newaxis,:]), axis=0)
			poseTemp, intrinsicTemp = getPoseIntrinsic(dataset_path2, i)
			pose1 = np.concatenate((pose1, poseTemp[np.newaxis,:]))
			intrinsic1 = np.concatenate((intrinsic1, intrinsicTemp[np.newaxis,:]))			

	depth1 = imageio.imread(os.path.join(dataset_path2, "depths", "{:04d}.exr".format(0)))
	depth1 = depth1[...,np.newaxis]

	rgb = np.stack((rgb0, rgb1))
	depth = np.stack((depth0, depth1))
	pose = np.stack((pose0, pose1))
	intrinsic = np.stack((intrinsic0, intrinsic1))

	return rgb, depth, pose, intrinsic

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

def getpose(pose):
	#poseInv = tf.matrix_determinant(pose)
	poseInv = tf.matrix_inverse(pose)
	return poseInv

if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES']='1'

	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	batch = 2
	frames = 3

	#define place holder
	rgbInput = tf.placeholder(tf.float32, [batch, frames, 480, 640, 3], name='rgbInput')
	depthInput = tf.placeholder(tf.float32, [batch, 480, 640, 1], name='depthInput')
	intrinsicInput = tf.placeholder(tf.float32, [batch, frames, 4], name='intrinsicInput')
	poseInput = tf.placeholder(tf.float32, [batch, frames, 4, 4], name='posesInput')

	poseGdtr = tf.placeholder(tf.float32, [batch, frames, 4, 4], name='posesInput')
	depthGdtr = tf.placeholder(tf.float32, [batch, 480, 640, 1], name='posesInput')
	
	#build network
	net_test = Network(rgbInput, poseInput, intrinsicInput, depthInput, poseGdtr, poseGdtr)
	pose_update  = net_test.build()

	#get image
	rgb, depth, pose, intrinsic = getImage()

	#test network
	sess.run(tf.global_variables_initializer())

	print "RUN!!!"

	result_pose = sess.run(pose_update, feed_dict={rgbInput: rgb, depthInput: depth, poseInput: pose, intrinsicInput: intrinsic})
	
	print "haha"
	for i in range(1):
		time_pre = time.time()
		result_pose = sess.run(pose_update, feed_dict={rgbInput: rgb, depthInput: depth, poseInput: pose, intrinsicInput: intrinsic})
		print time.time()-time_pre

	print result_pose.shape

	print result_pose