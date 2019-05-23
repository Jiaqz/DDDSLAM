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
#pose: [B,N-1,4,4]


os.environ['CUDA_VISIBLE_DEVICES']='0'

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

#build network

net_test = Network(rgbInput, poseInput, intrinsicInput, depthInput, poseGdtr, depthGdtr)
net_test.build()

output = net_test.get_output()
loss = net_test.get_pose_error()

global_step = tf.Variable(0, name='global_step', trainable=True)
lr = tf.train.exponential_decay(0.00001, global_step, 1000, 0.5, staircase=True)
optim = tf.train.AdamOptimizer(lr,beta1=0.5,beta2=0.9)
optim = tf.contrib.estimator.clip_gradients_by_norm(optim, 5.0)
train_step = optim.minimize(loss['Pose_pred_error'])
saver = tf.train.Saver(max_to_keep=100000)
#test network
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("log/", sess.graph)
merged = tf.summary.merge_all()




file = open('./testing_list.txt','r')
datalines = file.readlines()
size_data = len(datalines)
maxiter = 100000
index1 = 0
index2 = 0
lines1 = datalines[index1].split()
lines2 = datalines[index2].split()
rgb, depth, pose, intrinsic, pose_gt = getImage(lines1, lines2)

for i in range(maxiter):


    time_pre = time.time()

    Output, Loss, summary_tmp ,_= sess.run([output,loss,merged,train_step], 
    	feed_dict={rgbInput:rgb, poseInput:pose, intrinsicInput:intrinsic, depthInput:depth, poseGdtr:pose_gt, depthGdtr:depth})
    print 'iter', i, time.time()-time_pre, 'Loss ', Loss['Pose_pred_error'], 'Init_loss ', Loss['Pose_init_error']
    if i%10 == 0:
        writer.add_summary(summary=summary_tmp,global_step=i)
    #writer.add_summary(merged)

    if i%1000==0:
        saver.save(sess, './model/ckp', global_step=i)