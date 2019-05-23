import sys
import os
import time
import numpy as np
import imageio
import tensorflow as tf
from matplotlib import pyplot as plt
def getRefGradient(FeatureMaps, scope):
    with tf.variable_scope(scope):

        _, _, _, _, channel = FeatureMaps.get_shape().as_list()
        inputFeature = tf.transpose(FeatureMaps[:,0,:,:,:],(3,1,2,0))

        filter_x = tf.constant([[0,0,0],[-1,0,1],[0,0,0]], shape=[3, 3, 1, 1], dtype=tf.float32)
        filter_y = tf.constant([[0,-1,0],[0,0,0],[0,1,0]], shape=[3, 3, 1, 1], dtype=tf.float32)

        gradient_x = tf.transpose(tf.nn.conv2d(inputFeature, filter_x, strides=[1, 1, 1, 1], padding='SAME'),(3,1,2,0))
        gradient_y = tf.transpose(tf.nn.conv2d(inputFeature, filter_y, strides=[1, 1, 1, 1], padding='SAME'),(3,1,2,0))

    return gradient_x, gradient_y
img1 = (imageio.imread('/home/cjy/workspace/tensorflow/demon/datasets/mvs_test/0000/images/0000.png')).astype(np.float32)
img2 = (imageio.imread('/home/cjy/workspace/tensorflow/demon/datasets/mvs_test/0000/images/0001.png')).astype(np.float32)
imageio.imwrite('./img1.png',img1)
imageio.imwrite('./img2.png',img2)
img1 = img1[np.newaxis,:] / 255.0
img2 = img2[np.newaxis,:] / 255.0
img1_tensor = tf.constant(value = img1, dtype=tf.float32)
img2_tensor = tf.constant(value = img2, dtype=tf.float32)
input_img = tf.expand_dims(tf.concat((img1_tensor,img2_tensor),axis = 0),axis = 0)
grad_x, grad_y = getRefGradient(input_img,scope="Feature2_Gradient")
sess=tf.Session()
sess.run(tf.global_variables_initializer())
save_gx = (tf.reshape(grad_x[:,:,:,2],(480,640)).eval(session = sess))
save_gy = (tf.reshape(grad_y[:,:,:,2],(480,640)).eval(session = sess))
print np.max(save_gx)
print np.min(save_gx)

print np.max(save_gy)
print np.min(save_gy)

print np.sum(np.abs(save_gx)>0.3)
print np.sum(np.abs(save_gy)>0.3)
print np.sum(np.sqrt(save_gx**2+save_gy**2)>0.5)
print np.min(np.sqrt(save_gx**2+save_gy**2))
fig,axes = plt.subplots(1,2)
axes[0].imshow(save_gx)
axes[1].imshow(save_gy)
for j in range(2):
    axes[j].spines["top"].set_color("none")
    axes[j].spines["left"].set_color("none")
    axes[j].spines["right"].set_color("none")
    axes[j].spines["bottom"].set_color("none")
for ax in axes.flatten():
    ax.set_xticklabels([])
    ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)

fig.set_size_inches(8,3)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
fig.savefig('./grad.png',transparent=True,dpi=160)
plt.close()

save_gx[np.abs(save_gx)>0.3]=1.0
save_gx[save_gx != 1]=0
save_gy[np.abs(save_gy)>0.3]=1.0
save_gy[save_gy != 1]=0
fig,axes = plt.subplots(1,2)
axes[0].imshow(save_gx)
axes[1].imshow(save_gy)
for j in range(2):
    axes[j].spines["top"].set_color("none")
    axes[j].spines["left"].set_color("none")
    axes[j].spines["right"].set_color("none")
    axes[j].spines["bottom"].set_color("none")
for ax in axes.flatten():
    ax.set_xticklabels([])
    ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)

fig.set_size_inches(8,3)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
fig.savefig('./grad_high.png',transparent=True,dpi=160)
plt.close()

print grad_x.shape
print save_gx.shape
