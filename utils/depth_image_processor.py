import matplotlib.pyplot as plt
import numpy as np
import cv2

def depth_image_decode(depth_image_coded):
	"""
	Decode the depth image from the provided encoded image encoded_image_filename

	Parameters:
	encoded_image_filename

	Return:
	Return the decoded depth image with size same as input image file
	and also prints the data type of each entry in the image
	"""

	# Calculate the depth from the values at R,G and B channels using the following formula
	# R + G*256 + B*256*256
	depth_decoded=depth_image_coded[:,:,2]+depth_image_coded[:,:,1]*256 \
				  + depth_image_coded[:,:,0]*256*256

	# Normalize it in the range [0,1]
	depth_decoded=(depth_decoded/(256*256*256-1))

	# Convert the range from [0,1] to [0,255]
	depth_decoded=depth_decoded*255

	# print("Type of the data stored in coded depth array: %s" %depth_image_coded.dtype)
	#
	# print("Type of the data stored in decoded depth array: %s" %depth_decoded.dtype)

	return depth_decoded

def mask_depth_images(depth_image, limits):
	"""
	Mask (assign zeros / black color to) all the pixels whose values are
	not in the range mentioned

	Parameters:
	depth_image_decoded_array : 2D - array of the image
	[min, max] The range apart from which the pixels need to be masked

	Return:
	depth_image_masked_values : numpy array of size same as the input image
	 							but masked based on the range given
	"""

	# Apply the filter and create the mask as boolean
	depth_image_mask = (depth_image > limits[0]) & (depth_image < limits[1])

	# Apply the mask to the given image input
	depth_image_masked_values = depth_image * depth_image_mask

	return depth_image_masked_values

def log_scale_depth(depth_image):
	"""
	Convert the depth image to log scale

	Arguments
	depth_image : Depth Image with real depth value at each pixelself.

	Return:
	logdepth_image : depth image converted to the log scale
	"""

	# Log scale conversion is done based on the formula mentioned here :
	#  https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
	logdepth_image = (255/np.log(1+np.max(depth_image)))*np.log(1+depth_image)

	return logdepth_image

def load_depth(filename, log_scale_conversion='True'):
	"""
	Load the depth image --> Decode --> Scale Conversion --> Result

	Arguments:
	filename : name of the image file with the path
	log_scale_conversion : convert to the log scale or not (Default : True)

	Return:
	log_converted_depth : decoded and converted depth image
	"""
	depth_image_coded = cv2.imread(filename).astype(np.uint32)

	# Read and decode the depth image
	decoded_depth_image = depth_image_decode(depth_image_coded)

	# Convert the decoded depth image to the log scale
	log_converted_depth = log_scale_depth(decoded_depth_image)

	# Return the result image
	return log_converted_depth

def load_depth_generator(filename, height_range,
						 width_range, log_scale_conversion='True'):
	"""
	Load the depth image --> Decode --> Scale Conversion
	--> Return the compatible matrix array for the image generator

	Arguments:
	filename : name of the image file with the path
	log_scale_conversion : convert to the log scale or not (Default : True)
	width_range : tuple - (width_min, width_max)
	height_range : tuple - (height_min, height_max)

	Return:
	result_image : decoded and converted depth image generator compatible
	"""

	log_converted_depth = load_depth(filename, log_scale_conversion)

	# Crop only the required region
	log_converted_depth = log_converted_depth[height_range[0]:height_range[1],
											  width_range[0]:width_range[1]]

	log_converted_depth = log_converted_depth.reshape(
									log_converted_depth.shape[0],
									log_converted_depth.shape[1],
									1)
	result_image = np.expand_dims(log_converted_depth,axis=0)

	return result_image
