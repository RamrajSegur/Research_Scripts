import matplotlib.image as mpimg
import numpy as np
import cv2

def reshape(label_image, scale):
    """
    Reshape the images based on the scale

    Arguments:
    label_image : image matrix in the form of numpy array
    scale: the scale to which image has to be converted

    Return:
    result_image : image matrix rescaled
    """
    # Know the width and height of the image
    height, width = label_image.shape

    # Resize the image according to the desired scale
    result_image = cv2.resize(label_image,
                             dsize=((width//32)*scale,(height//32)*scale),
                             interpolation = cv2.INTER_NEAREST)
    return result_image

def load_label(filename, height_range , width_range, scale = None):
    """
    Load the image and return as a numpy matrix

    Arguments:
    filename : name of the image file
    width_range : width_range of the image required
    height_range : height_range of the image required
    scale : scale to which the image can be resized (Default : None)

    Return:
    label_image : return the cropped and scaled(if applied) image numpy matrix
    """
    # Read the image as the numpy matrix (values range from 0 to 1)
    label_image = mpimg.imread(filename)

    # Convert the values range from [0,1] to [0,255]
    label_image = label_image[:,:,0]*255

    # Resize if the scale parameter is given
    if scale == None:
        label_image = label_image[height_range[0]:height_range[1],
                                  width_range[0]:width_range[1]]
    else:
        label_image = label_image[height_range[0]:height_range[1],
                                  width_range[0]:width_range[1]]
        label_image = reshape(label_image, scale)


    return label_image

def load_label_generator(filename, width_range, height_range, scale = None):
    """
    Load the image, crop it, scale it and convert to compatible with generaor

    Arguments:
    filename : name of the image file
    width_range : width_range of the image required
    height_range : height_range of the image required
    scale : scale to which the image can be resized (Default : None)

    Return:
    label_image : return the cropped and scaled(if applied) image numpy matrix
                  and compatible to generator
    """
    # Load the image, crop and resize accordingly
    label_image = load_label(filename, width_range, height_range, scale)

    # Convert the label_image to int type
    label_image = label_image.astype(int)

    # Initialize img_coded array with zeros
    img_coded=np.zeros_like(label_image)

    # Encode the layer information to img_coded file based on underlying label
    for i in range((label_image.shape[0])):
        for j in range((label_image.shape[1])):
            if label_image[i][j]==10.0:
                img_coded[i][j]=2

            elif label_image[i][j]==7.0:
                img_coded[i][j]=1

            else:
                img_coded[i][j]=0
    # Ensure the datatype as comparable using the if statements
    img_coded = np.array(img_coded, dtype=np.uint8)

    # Create the result array as full of zeros
    y=np.zeros((1,label_image.shape[0],label_image.shape[1],3),dtype=np.float32)

    # Encode them based on the layer information of each pixel
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            y[0,i,j,img_coded[i][j]]=1
    return y
