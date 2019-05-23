import tensorflow as tf
from SE3 import *
#General case of bilinear sampling
#imgs: [B, H, W, C]
#coords: [B, H, W, 2]
def bilinearSampler(imgs, coords):

	with tf.variable_scope("BilinearSampler"):
		batch, height, width, _ = imgs.get_shape().as_list()
	
		coords_i = tf.reshape(tf.range(batch), [batch, 1, 1, 1])
		coords_i = tf.tile(coords_i, [1, height, width, 1])
		coords_i = tf.cast(coords_i, 'float32')
	
		coords_x = coords[..., :1]
		coords_y = coords[..., 1:]
	
		coords_x = tf.where(tf.is_nan(coords_x), tf.zeros_like(coords_x), coords_x)
		coords_y = tf.where(tf.is_nan(coords_y), tf.zeros_like(coords_y), coords_y)
	
		mask = tf.to_float( (coords_x > 0)&(coords_x < width-1)&(coords_y > 0)&(coords_y < height-1) )
		mask = tf.expand_dims(mask, axis=-1)
	
		coords_x = tf.clip_by_value(coords_x, 0, width-1)
		coords_y = tf.clip_by_value(coords_y, 0, height-1)
	
		x0 = tf.floor(coords_x)
		x1 = x0 + 1
		y0 = tf.floor(coords_y)
		y1 = y0 + 1
	
		x0 = tf.cast(x0, 'int32')
		x1 = tf.cast(x1, 'int32')
		y0 = tf.cast(y0, 'int32')
		y1 = tf.cast(y1, 'int32')
		coords_i = tf.cast(coords_i, 'int32')
	
		coords00 = tf.stack([coords_i, y0, x0], axis=-1)
		coords01 = tf.stack([coords_i, y0, x1], axis=-1)
		coords10 = tf.stack([coords_i, y1, x0], axis=-1)
		coords11 = tf.stack([coords_i, y1, x1], axis=-1)
	
		img00 = tf.gather_nd(imgs, coords00)
		img01 = tf.gather_nd(imgs, coords01)
		img10 = tf.gather_nd(imgs, coords10)
		img11 = tf.gather_nd(imgs, coords11)
	
		dx = coords_x - tf.cast(x0, 'float32')
		dy = coords_y - tf.cast(y0, 'float32')
		dx = tf.expand_dims(dx, axis=-1)
		dy = tf.expand_dims(dy, axis=-1)
	
		w00 = (1.0 - dy) * (1.0 - dx)
		w01 = (1.0 - dy) * dx
		w10 = dy * (1.0 - dx)
		w11 = dy * dx
	
		output = mask * tf.add_n([w00 * img00, w01 * img01, w10 * img10, w11 * img11, ])
	
	return output, mask


#imgs: [B, N, H, W, C]
#coords: [B, N, H, W, 2]
def bilinearSamplerMultiframe(imgs, coords):

	with tf.variable_scope("BilinearSamplerMultiframe"):
		batch, frame, height, width, channel = imgs.get_shape().as_list()
	
		imgReshape = tf.reshape(imgs, [batch*frame, height, width, channel])
		coordReshape = tf.reshape(coords, [batch*frame, height, width, coords.shape[-1]])
	
		outputReshape, maskReshape = bilinearSampler(imgReshape, coordReshape)
	
		outputs = tf.reshape(outputReshape, [batch, frame, height, width, channel])
		masks = tf.reshape(maskReshape, [batch, frame, height, width, 1])

	return outputs, masks


def coordGrid(batch, height, width, homogeneous=False):

	with tf.variable_scope("CoordGrid"):
		cam_coords = tf.meshgrid(tf.range(width), tf.range(height))
		coords = tf.stack(cam_coords, axis=-1)
		coords = tf.to_float(tf.expand_dims(coords, 0))
	
	if homogeneous:
		coords = tf.concat([coords, tf.ones([1,height,width,1])], axis=-1)
	
		coords = tf.tile(coords, [batch, 1, 1, 1])
	return coords


#depth [B, H, W, 1]
#intrinsic [B, 4] fx, fy, cx, cy
def pointCloudFromDepth(depths, intrinsics):

	with tf.variable_scope("PointCloudFromDepth"):
		batch, height, width, _ = depths.get_shape().as_list()
		coords = coordGrid(batch, height, width)												#[B, H, W, 2] 2:u, v
		intrinsicExpand = tf.expand_dims(tf.expand_dims(intrinsics, 1), 1)
		intrinsicExpand = tf.tile(intrinsicExpand, [1, height, width, 1]) 						#[B, H, W, 4]
	
		fx, fy, cx, cy = tf.split(intrinsicExpand, [1, 1, 1, 1], axis=-1)						#[B, H, W, 1]
	
		X = (coords[:,:,:,:1] - cx) / fx * depths
		Y = (coords[:,:,:,1:] - cy) / fy * depths

	return tf.concat([X, Y, depths], -1)													#[B, H, W, 3] 3:X,Y,Z


#pointClouds [B, H, W, 3]
#poses [B, N, 4, 4] 4,4:SE3
#pointCloudTrans [B, N, H, W, 3] 3:xyz
def pointCloudTransformation(pointClouds, poses):
	
	with tf.variable_scope("PointCloudTransformation"):
		batch, height, width, _ = pointClouds.get_shape().as_list()
		_, frame, _, _ = poses.get_shape().as_list()
	
		pointCloudExpand = tf.expand_dims(pointClouds, 1)
		pointCloudExpand = tf.tile(pointCloudExpand, [1, frame, 1, 1, 1])							#[B, N, H, W, 3]
	
		pointCloudhomoge = tf.tile(tf.constant(1.0,shape=[1, 1, 1, 1, 1]), [batch, frame, height, width, 1])
		pointCloudExpand = tf.concat([pointCloudExpand, pointCloudhomoge], -1)						#[B, N, H, W, 4]
	
		poseExpand = tf.expand_dims(tf.expand_dims(poses, 2),2)											#[B, N, 1, 1, 4, 4]
		poseExpand = tf.tile(poseExpand, [1, 1, height, width, 1, 1])
	
		pointCloudExpandReshape = tf.reshape(pointCloudExpand, [batch*frame*height*width, pointCloudExpand.shape[-1]])
		pointCloudExpandReshape = tf.expand_dims(pointCloudExpandReshape, -1)
		poseExpandReshape = tf.reshape(poseExpand, [batch*frame*height*width, poseExpand.shape[-2], poseExpand.shape[-1]])
	
		pointCloudTransExpandReshape = tf.einsum('aij,ajk->aik', poseExpandReshape, pointCloudExpandReshape)
		pointCloudTransExpandReshape = pointCloudTransExpandReshape[...,0]
		pointCloudTransExpand = tf.reshape(pointCloudTransExpandReshape, [batch, frame, height, width, pointCloudTransExpandReshape.shape[-1]])
		pointCloudTrans = pointCloudTransExpand[:,:,:,:,:3]

	return pointCloudTrans


#pointClouds [B, N, H, W, 3] 3:x, y, z
#intrinsic [B, N, 4]
#coord [B, N, H, W, 3] 3:u,v,z
def projectFromPointCloud(pointClouds, intrinsics):
	
	with tf.variable_scope("ProjectFromPointCloud"):
		batch, frame, height, width, _ = pointClouds.get_shape().as_list()
	
		intrinsicExpand = tf.expand_dims(tf.expand_dims(intrinsics, 2), 2)
		intrinsicExpand = tf.tile(intrinsicExpand, [1, 1, height, width, 1])
		fx, fy, cx, cy = tf.split(intrinsicExpand, [1, 1, 1, 1], axis=-1)
	
		fx = fx[..., 0]
		fy = fy[..., 0]
		cx = cx[..., 0]
		cy = cy[..., 0]
	
		u = pointClouds[..., 0] / pointClouds[..., 2] * fx + cx
		v = pointClouds[..., 1] / pointClouds[..., 2] * fy + cy
		#z = pointClouds[..., 2]

	return tf.stack([u,v], axis=-1)


#depths [B, H, W, 1]
#poses [B, N-1, 4, 4]
#intrinsics [B, N, 4]
#coord [B, N-1, H, W, 3]
def reprojection(depths, poses, intrinsics):
	
	with tf.variable_scope("Reprojection"):
		refIntrinsics = intrinsics[:, 0, :]
		neiIntrinsics = intrinsics[:, 1:, :]
	
		pointClouds = pointCloudFromDepth(depths, refIntrinsics)
		pointCloudsTrans = pointCloudTransformation(pointClouds, poses)
		coord = projectFromPointCloud(pointCloudsTrans, neiIntrinsics)

	return coord, pointCloudsTrans


#Gx: [B, N-1, H, W, C]
#Gy: [B, N-1, H, W, C]
#Point3d: [B, N-1, H, W, 3]
#Intrinsics: [B, N-1, 4]
#J [B*(N-1)*H*W, C, 6]
def computeJacobian(Gx, Gy, Point3d, Intrinsics):

	with tf.variable_scope("ComputeJacobian"):
		batch, frames, height, width, channel = Gx.get_shape().as_list()
		M = batch * frames
		HW = height * width
		MHW = M * HW
		Mask_high_grad = tf.to_float(tf.sqrt(Gx**2+Gy**2)>-0.1)
		Gx = tf.reshape(Gx, (MHW, channel, 1))
		Gy = tf.reshape(Gy, (MHW, channel, 1))
		Point3d = tf.reshape(Point3d, (MHW, 3))
		Intrinsics = tf.reshape(Intrinsics, (M, 4))

		#Gxy: MHW*channel*2
		Gxy = tf.concat((Gx, Gy), axis=2)
		
		Intrinsics = tf.reshape(tf.tile(Intrinsics, (1, HW)), (MHW, 4))
		
		pX, pY, pZ = tf.split(Point3d, [1, 1, 1], axis=-1)
		fx, fy, cx, cy = tf.split(Intrinsics, [1, 1, 1, 1], axis=-1)
		
		fx = tf.reshape(fx, [-1, 1])
		fy = tf.reshape(fy, [-1, 1])
		cx = tf.reshape(cx, [-1, 1])
		cy = tf.reshape(cy, [-1, 1])
	
		zero = tf.zeros_like(pZ)
		ones = tf.ones_like(pZ)
		d = 1.0 / pZ
		
		#J_2: MHW*2*6
		J_2 = tf.stack([
			tf.concat([fx*d, zero, -fx*pX*d**2, -fx*pX*pY*d**2, fx*(1+pX**2*d**2), -fx*pY*d], axis=-1),
			tf.concat([zero, fy*d, -fy*pY*d**2, -fy*(1+pY**2*d**2), fy*pX*pY*d**2,  fy*pX*d], axis=-1),
		], axis=-2)
		
		#J: MHW*channel*6
		J = -tf.matmul(Gxy, J_2)
	
	return J,Mask_high_grad


def getDeltaPose(J, Ex, Mask, Mask_High_Grad):
	'''
		J: MHW*channel*6
		Ex: B*(N-1)*H*W*C
		Mask: B*(N-1)*H*W*1
	
		return:
		    xi: M*6
	'''
	with tf.variable_scope("getDeltaPose"):

		batch, frames, height, width, channel = Ex.get_shape().as_list()
		M = batch * frames
		HW = height * width
		MHW = M * HW

		J = tf.reshape(J, (M, HW*channel, 6))
		Mask = Mask*Mask_High_Grad
		Mask = tf.tile(tf.reshape(Mask, (M, HW, 1)), [1,channel,1])

		JT = tf.transpose(J, (0, 2, 1))
		JT_J = tf.matmul(JT, J*Mask)

		eye = 0.1*tf.eye(6, batch_shape=[M])

		JT_J_inv = tf.matrix_inverse(JT_J + eye)

		Ex = tf.reshape(Ex, (M, HW*channel, 1))
		b = -tf.matmul(JT, Ex*Mask)

		xi = tf.matmul(JT_J_inv, b)

		xi = tf.reshape(xi, (M, 6))

	return xi

#Gx: [B, N-1, H, W, C]
#Gy: [B, N-1, H, W, C]
#depth: [B, H, W, 1] 
#inputIntrinsics: [B, N-1, 4]
#poseRelative: [B, N-1, 4, 4]
def poseUpdate(Gx, Gy, depth, feature_Ref, feature_Nei, poseRelative, inputIntrinsics):

	with tf.variable_scope("poseUpdate"):

		batch, frames, height, width, channel = Gx.get_shape().as_list()

		coord, pointCloud = reprojection(depth, poseRelative, inputIntrinsics)

		gradient_x_sample, mask = bilinearSamplerMultiframe(Gx, coord)
		gradient_y_sample, _    = bilinearSamplerMultiframe(Gy, coord)

		feature_sample, _ = bilinearSamplerMultiframe(feature_Nei, coord)
		featureRefExtra = tf.tile(feature_Ref, [1, frames, 1, 1, 1])
		featureError = featureRefExtra - feature_sample

		Jacob, Mask_high_grad = computeJacobian(gradient_x_sample, gradient_y_sample, pointCloud, inputIntrinsics)
		deltaPose = getDeltaPose(Jacob, featureError, mask, Mask_high_grad)

		deltaPose = tf.clip_by_value(deltaPose, -0.1, 0.1)

		poseRelative = tf.reshape(poseRelative, (batch*frames, 4, 4))

		G_update = increment(poseRelative, deltaPose)

		G_update = tf.reshape(G_update, (batch, frames, 4, 4))

	return G_update