import numpy as np
import os
import matplotlib.pyplot as plt

from image import Image


if __name__ == '__main__':
	# get list of image names
	im_list = [file for file in os.listdir('./imgs') if file.endswith('.jpg')]

	# for each image, instantiate an Image object to calculate the Homography that map points from plane to image
	images = [Image(os.path.join('./imgs', im_file)) for im_file in im_list]

	# TODO: construct V to solve for b by stacking the output of im.construct_v() (Equation.(17))
	V = np.vstack([im.construct_v() for im in images])

	# TODO: find b using the SVD trick
	_, _, Vt = np.linalg.svd(V)
	b = Vt[-1]
	print('norm(V @ b) =', np.linalg.norm(V @ b))  # check if the dot product between V and b is zero
	b11, b12, b22, b13, b23, b33 = b.tolist()
	print('b.shape: ', b.shape)
	print('b11: ', b11)
	print('b12: ', b12)
	print('b22: ', b22)
	print('b13: ', b13)
	print('b23: ', b23)
	print('b33: ', b33)

	# TODO: find components of intrinsic matrix from Equation.(12)
	v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)

	lam = b33 - (b13**2 + v0 * (b12 * b13 - b11 * b23)) / b11

	alpha = np.sqrt(lam / b11)
	beta = np.sqrt(lam * b11 / (b11 * b22 - b12**2))
	c = -b12 * alpha**2 * beta / lam
	u0 = c * v0 / beta - b13 * alpha**2 / lam

	print('----\nCamera intrinsic parameters:')
	print('\talpha: ', alpha)
	print('\tbeta: ', beta)
	print('\tc: ', c)
	print('\tu0: ', u0)
	print('\tv0: ', v0)
	cam_intrinsic = np.array([
		[alpha, c, u0],
		[0, beta, v0],
		[0, 0, 1]
	])

	# get camera pose
	for im in images:
		R, t = im.find_extrinsic(cam_intrinsic)
		print('R = \n', R)
		# print('R @ R.T = \n', R @ R.T)
		print('t = ', t)
		
		landmark_in_world = im.get_landmark_world_coordinate()  # (N, 2)

		proj = np.stack([
			R[:, 0], R[:, 1], t
		], axis=1)

		landmark_in_world = np.pad(landmark_in_world, pad_width=[(0, 0), (0, 1)], constant_values=0)  # (N, 3)

		landmark_in_world = landmark_in_world @ R.T + t

		landmark_in_pixel = landmark_in_world @ cam_intrinsic.T

		landmark_in_pixel /= landmark_in_pixel[:, [-1]]

		fig, ax = plt.subplots()
		ax.imshow(im.im[..., ::-1])
		ax.scatter(landmark_in_pixel[:, 0], landmark_in_pixel[:, 1])

	
	plt.show()
