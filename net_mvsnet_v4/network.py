import sys
import os
import time
import numpy as np

import tensorflow as tf

from geometry import *
from SE3 import *
DEFAULT_PADDING = 'SAME'
def pose_perturbation(G, delta):
	G = tf.reshape(G,(2,4,4))
	xi_dim = [2,6]
	#dxi = tf.random_normal(xi_dim)
	dxi = tf.constant([[-0.67911141, -0.0082109 , -0.6949065 , -0.59913895,  0.66489155,-1.16384883],[-0.67911141, -0.0082109 , -0.6949065 , -0.59913895,  0.66489155,-1.16384883]])
	G1 = increment(G, dxi*delta)
	G1 = tf.reshape(G1,(1,2,4,4))
	return G1
def leaky_relu(x, alpha=0.2):
	with tf.variable_scope("leaky_relu"):
		result = tf.maximum(tf.minimum(0.0, alpha * x), x)
	return result

class Network(object):

	def __init__(self, inputImages, inputPoses, inputIntrinsics, inputDepth, gdtrPoses, gdtrDepth, is_training=True, reuse=False):
		
		self.is_training = is_training
		self.reuse = reuse
		
		self.inputImages = inputImages
		self.inputPoses = inputPoses
		self.inputIntrinsics = inputIntrinsics

		self.inputDepth = inputDepth
		
		self.gdtrPoses = gdtrPoses
		self.gdtrDepth = gdtrDepth

		self.output = {}
		self.pose_error = {}


	def build(self):

		#get feature ----------------------------------------------------------------------------------
		feature1 = self.featureExtractBlock(self.inputImages)
		feature1 = tf.expand_dims(tf.reduce_mean(feature1,axis=-1),axis=-1)
		tf.summary.image('image_0',self.inputImages[0,1:2,:,:,:1])

		with tf.variable_scope("FeatureRef"):
			feature1Ref = feature1[:,:1]

		with tf.variable_scope("FeatureNei"):
			feature1Nei = feature1[:,1:]
			tf.summary.image('Feature_x_1',feature1Nei[0,:1,:,:,:1])
		
		## pose part ----------------------------------------------------------------------------------
		#get feature gradient
		#FeatureMaps: [B, N-1, H, W, C]
		#grandient: [B, N-1, H, W, C]
		with tf.variable_scope("FeatureGradient"):
			gradient1_x, gradient1_y = self.getRefGradient(feature1Nei, scope="Feature1_Gradient")

			tf.summary.image('Gx_1',gradient1_x[0,:1,:,:,:1])

		#get warp coordinate
		#depths [B, H, W, 1] each image sequeence only have one reference depth image
		#poses [B, N, 4, 4]
		#intrinsics [B, N, 4]
		#coord [B, N-1, H, W, 3] only need coordinate and point cloud of neiboring image
		high_grad = tf.to_float(tf.sqrt(gradient1_x**2+gradient1_y**2)>0.1)
		tf.summary.image('high_grad',high_grad[0,:1,:,:,:1])
		high_grad_size = tf.reduce_sum(high_grad)
		tf.summary.scalar("high_grad_size", high_grad_size)
		with tf.variable_scope("DepthResize"):
			depth1 = tf.image.resize_images(self.inputDepth, [self.inputDepth.shape[1] / 4, self.inputDepth.shape[2] / 4], method=tf.image.ResizeMethod.BILINEAR)

		#init pose
		with tf.variable_scope("Pose"):
			poseRef = self.inputPoses[:,:1]
			poseNei = self.inputPoses[:,1:]
			poseRefExtra = tf.tile(poseRef, [1, poseNei.shape[1], 1, 1])

		with tf.variable_scope("PoseRelative"):
			batch, frame, height, width = poseNei.get_shape().as_list()
			poseNeiReshape = tf.reshape(poseNei, [batch*frame, height, width])
			poseRefExtraReshape = tf.reshape(poseRefExtra, [batch*frame, height, width])
			poseRelative = tf.reshape(tf.einsum('aij,ajk->aik', poseNeiReshape, inv_SE3(poseRefExtraReshape)), [batch, frame, height, width])

		#pose update iteratively
		#60*80
		delta = tf.tile(tf.reshape(tf.constant([0.025, 0.025, 0.025, 0.025, 0.025, 0.025]),(1,6)),[2,1])
		poseRelative_update = pose_perturbation(poseRelative,delta) 

		self.output['Pose_init'] = poseRelative_update

		#120*160
		with tf.variable_scope("updataPose120160"):
			for i in range(9):
				with tf.variable_scope(str(i)):
					poseRelative_update = poseUpdate(gradient1_x, gradient1_y, depth1, feature1Ref, feature1Nei, poseRelative_update, self.inputIntrinsics[:,1:,:]/4)
		
		self.output['Pose_pred'] = poseRelative_update
		## depth part ---------------------------------------------------------------------------------

		#warp feature to build cost volume

		#convolution of them

		#result & loss

	# input images: [B, N, H, W, C ]
	# B: batch size
	# N: each input image group has N images, N0 is reference image, and N1~Nn is neiborhood image (N could be 1 when tracking each image)
	# H: image hight
	# W: image width
	# C: image channel, for RGB image, C = 3
	def conv(self,
			input,
			kernel_size,
			filters,
			strides,
			name,
			relu=True,
			padding=DEFAULT_PADDING,
			biased=False):
		kwargs = {'filters': filters,
				'kernel_size': kernel_size,
				'strides': strides,
				'activation': tf.nn.relu if relu else None,
				'use_bias': biased,
				'padding': padding,
				'trainable': self.is_training,
				'reuse': self.reuse,
				'name': name}
		if len(input.get_shape()) == 4:
			return tf.layers.conv2d(input, **kwargs)
		else:
			raise ValueError('Improper input rank for layer: ' + name)
	def conv_bn(self,
				input,
				kernel_size,
				filters,
				strides,
				name,
				relu=True,
				center=False,
				padding=DEFAULT_PADDING):
		kwargs = {'filters': filters,
				'kernel_size': kernel_size,
				'strides': strides,
				'activation': None,
				'use_bias': False,
				'padding': padding,
				'trainable': self.is_training,
				'reuse': self.reuse}

		with tf.variable_scope(name):
			if len(input.get_shape()) == 4:
				conv = tf.layers.conv2d(input, **kwargs)
			else:
				raise ValueError('Improper input rank for layer: ' + name)
			# note that offset is disabled in default
			# scale is typically unnecessary if next layer is relu.
			output = tf.layers.batch_normalization(conv,
														center=center,
														scale=False,
														training=self.is_training,
														fused=True,
														trainable=center,
														reuse=self.reuse)
			if relu:
				output = tf.nn.relu(output)
			return output
	def featureExtractBlock(self, inputImages, reuse=False):
		with tf.variable_scope("ExtractFeatureBlock"):

			batch, frames, height, width, channel = inputImages.get_shape().as_list()
			inputImagesReshaped = tf.reshape(inputImages, [batch*frames, height, width, channel])

			base_filter = 8
			x0 = self.conv_bn(inputImagesReshaped, 3, base_filter, 1, name='conv0_0')
			x0 = self.conv_bn(x0, 3, base_filter, 1, name='conv0_1')
			x0 = self.conv_bn(x0, 5, base_filter * 2, 2, name='conv1_0')
			x0 = self.conv_bn(x0, 3, base_filter * 2, 1, name='conv1_1')
			x0 = self.conv_bn(x0, 3, base_filter * 2, 1, name='conv1_2')
			x0 = self.conv_bn(x0, 5, base_filter * 4, 2, name='conv2_0')
			x0 = self.conv_bn(x0, 3, base_filter * 4, 1, name='conv2_1')
			x0 = self.conv(x0, 3, base_filter * 4, 1, relu = False, name='conv2_2')

			return tf.reshape(x0, [batch, frames, height/4, width/4, x0.shape[3]])

	# FeatureMaps: [B, N, H, W, C]
	#only calculate the feature gradient of reference image feature
	def getRefGradient(self, featureMaps, scope):
		
		with tf.variable_scope(scope):

			batch, frame, height, width, channel = featureMaps.get_shape().as_list()
			featureMapTrans = tf.transpose(featureMaps,(0, 1, 4, 2, 3))		#[B, N, H, W, C] -> [B, N, C, H, W]
			featureMapTransReshape = tf.expand_dims(tf.reshape(featureMapTrans, [batch * frame * channel, height, width]), -1) #[B*N*C, H, W, 1]

			fillterX = tf.constant([[0,0,0],[-1,0,1],[0,0,0]], shape=[3, 3, 1, 1], dtype=tf.float32)
			fillterY = tf.constant([[0,-1,0],[0,0,0],[0,1,0]], shape=[3, 3, 1, 1], dtype=tf.float32)

			gradientXTransReshape = tf.nn.conv2d(featureMapTransReshape, fillterX, strides=[1, 1, 1, 1], padding='SAME')#[...,0]
			gradientYTransReshape = tf.nn.conv2d(featureMapTransReshape, fillterY, strides=[1, 1, 1, 1], padding='SAME')#[...,0]

			gradientXTrans = tf.reshape(gradientXTransReshape, [batch, frame, channel, height, width])
			gradientYTrans = tf.reshape(gradientYTransReshape, [batch, frame, channel, height, width])
			gradientX = tf.transpose(gradientXTrans, [0, 1, 3, 4, 2])
			gradientY = tf.transpose(gradientYTrans, [0, 1, 3, 4, 2])

		return gradientX, gradientY



	def compute_loss(self, pose_star, pose_pred, scope):

		with tf.variable_scope('computeLoss_'+scope):

			R1, t1 = pose_star[:,:, 0:3, 0:3], pose_star[:,:, 0:3, 3]
			R2, t2 = pose_pred[:,:, 0:3, 0:3], pose_pred[:,:, 0:3, 3]

			ri = tf.trace(tf.matmul(tf.transpose(R2, [0, 1, 3, 2]), R1))
			angle = tf.acos(tf.minimum(1.0, tf.maximum(-1.0, (ri-1)/2)))
			rotation_error = tf.reduce_mean(tf.abs(angle))

			translation_error = tf.reduce_mean(tf.abs(t1-t2))

			motion_loss = translation_error+rotation_error

			tf.summary.scalar("rotation_error", rotation_error)
			tf.summary.scalar("translation_error", translation_error)
			tf.summary.scalar("motion_error", motion_loss)
		
		return motion_loss

	def get_output(self):
		return self.output

	def get_pose_error(self):
		pose_init = self.output['Pose_init']
		pose_pred = self.output['Pose_pred']
		self.pose_error['Pose_init_error'] = self.compute_loss(self.gdtrPoses,pose_init,'init')
		self.pose_error['Pose_pred_error'] = self.compute_loss(self.gdtrPoses,pose_pred,'pred')
		return self.pose_error


