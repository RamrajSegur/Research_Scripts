from keras.backend import tf as ktf

def tensor_resize_by_scale(tensor, scale, upsample = 'False'):
    """
    Resize the tensor according to the scale and upsample preference

    Arguments:
    tensor : tensor object to be resized
    scale : amount by which the tensor has to be reshaped
    upsample : if the scale has to magnify or reduce the size

    Return:
    result_tensor : tensor as a result of the scaling and resizing operation
    """
    if upsample == 'False':
        result_tensor = ktf.image.resize_nearest_neighbor(tensor,
                    size=(int(tensor.shape[1])//scale,
                          int(tensor.shape[2])//scale))
    else:
        result_tensor = ktf.image.resize_nearest_neighbor(tensor,
                    size=(int(tensor.shape[1])*scale,
                          int(tensor.shape[2])*scale))
    return result_tensor

def tensor_resize_by_value(tensor, value):
    """
    Resize the tensor according to the value of size

    Arguments:
    tensor : tensor object to be resized
    value : list having sizes for height and width [height_size , width_size]

    Return:
    result_tensor : tensor as a result of resizing operation
    """

    result_tensor =ktf.image.resize_nearest_neighbor(tensor,size=value)

    return result_tensor
