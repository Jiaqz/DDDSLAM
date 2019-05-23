import os
import sys
import numpy as np
import imageio
import tensorflow as tf
import json
import time
import matplotlib.pyplot as plt
import random
from network import Network
from data import *

os.environ['CUDA_VISIBLE_DEVICES']='1'
gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
batch = 1
frames = 3

#define place holder
rgbInput = tf.placeholder(tf.float32, [batch, frames, 480, 640, 3], name='rgbInput')
depthInput = tf.placeholder(tf.float32, [batch, 480, 640, 1], name='depthInput')
intrinsicInput = tf.placeholder(tf.float32, [batch, frames, 4], name='intrinsicInput')
poseInput = tf.placeholder(tf.float32, [batch, frames, 4, 4], name='poseInput')

poseGdtr = tf.placeholder(tf.float32, [batch, frames-1, 4, 4], name='poseGdtr')
posePred = tf.placeholder(tf.float32, [batch, frames-1, 4, 4], name='posePred')
depthGdtr = tf.placeholder(tf.float32, [batch, 480, 640, 1], name='depthGdtr')

net_test = Network(rgbInput, poseInput, intrinsicInput, depthInput, poseGdtr, depthGdtr,False)
net_test.build()
output = net_test.get_output()
loss = net_test.get_pose_error()
saver = tf.train.Saver(tf.global_variables())
#saver = tf.train.Saver()
saver.restore(sess, './model/ckp-0')


file = open('./testing_list.txt','r')
datalines = file.readlines()
size_data = len(datalines)
maxiter = 100000
index1 = 0
index2 = 0
lines1 = datalines[index1].split()
lines2 = datalines[index2].split()
rgb, depth, pose, intrinsic, pose_gt = getImage(lines1, lines2)
Output, Loss = sess.run([output,loss], feed_dict={rgbInput:rgb, poseInput:pose, intrinsicInput:intrinsic, depthInput:depth, poseGdtr:pose_gt, depthGdtr:depth})
print 'Loss ', Loss['Pose_pred_error'], 'Init_loss ', Loss['Pose_init_error']
