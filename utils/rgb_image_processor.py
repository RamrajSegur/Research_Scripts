import numpy as np
import matplotlib.image as mpimg

def load_rgb(filename,width_range, height_range):
    """
    Read the file and return the image matrix (Values from 0 to 1)

    Arguments:
    filename : name of the rgb image with the path
    width_range : a tuple (width_min, width_max)
    height_range : a tuple (height_min, height_max)

    Returns:
    image_matrix : returns the numpy matrix of the image
    """
    # Read the image file
    image_matrix = mpimg.imread(filename)

    #Check if the image file is having three channels else raise error
    if image_matrix.shape[2]==3:
        image_matrix = image_matrix[width_range[0]:width_range[1],
                                    height_range[0]:height_range[1]]
    else:
        raise ValueError("The number of channels in the image is != 3")

    return image_matrix


def load_rgb_generator(filename,width_range, height_range):
    """
    Read the file and return the image matrix (Values from 0 to 1) and
    resized to be compatible to the generator

    Arguments:
    filename : name of the rgb image with the path
    width_range : a tuple (width_min, width_max)
    height_range : a tuple (height_min, height_max)

    Returns:
    image_matrix : returns the numpy matrix of the image
    """
    # Get the image matrix correspoding to the image filename
    image_matrix = load_rgb(filename, width_range, height_range)

    # Reshape the image to be compatible for the generator
    image_matrix_gen = np.expand_dims(image_matrix, axis=0)

    return image_matrix_gen
