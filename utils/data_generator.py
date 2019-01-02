import os
from utils import depth_image_processor as dip
from utils import rgb_image_processor as rip
from utils import label_image_processor as lip

def generate_arrays_from_file(file, image_dir, label_dir, depth_dir,
                              height_range, width_range):

    """
    Generate arrays from the file containing filenames

    Arguments:
    file : the file containing the filenames
    image_dir : path for the folder containing RGB images
    label_dir : path for the folder containing Label images
    depth_dir : path for the folder containing depth images
    height_range : height_range of the image required
    width_range : width_range of the image required

    Return:
    Return the list of arrays needed for each training step
    """

    while 1:
        f = open(file)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename)
            path_label = os.path.join(label_dir, filename)
            path_depth = os.path.join(depth_dir, filename)
            x = rip.load_rgb_generator(path_image,height_range,width_range)
            depth = dip.load_depth_generator(path_depth,height_range,
                                             width_range)
            w = lip.load_label_generator(path_label,height_range,width_range,8)
            y = lip.load_label_generator(path_label,height_range,width_range,4)
            z = lip.load_label_generator(path_label,height_range,width_range,2)
            yield ([x,depth], [w, y, z])
        f.close()
