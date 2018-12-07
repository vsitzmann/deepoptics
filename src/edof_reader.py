import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import os

from glob import glob

def get_edof_training_data(target_dir,
                           patch_size,
                           batch_size,
                           num_depths=4,
                           num_par_calls=4,
                           repeat=True):
    '''Data reader that ingests images and cuts out patches with patch_size.

    :param target_dir (str): Directory containing images.
    :param patch_size (str): Size of the patches to be extracted from images.
    :param batch_size: Batch size.
    :param num_depths: Number of depth planes that a single image is placed at.
    :param num_par_calls: Number of parallel readers.
    :param repeat: Whether to loop the dataset.
    :return:
    '''
    def parse_img(img_path):
        img_string = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img = tf.image.convert_image_dtype(img_decoded, tf.float32)
        return img

    def augment_img(img):
        return tf.image.resize_image_with_crop_or_pad(
            img,
            patch_size,
            patch_size
        )

    img_path_list = glob(os.path.join(target_dir, '*.jpg')) + glob(os.path.join(target_dir, '*.png'))
    img_dataset = Dataset.from_tensor_slices((img_path_list))
    img_dataset = img_dataset.shuffle(1000)

    if repeat:
        img_dataset = img_dataset.repeat()

    # Read the images
    img_dataset = img_dataset.map(parse_img, num_parallel_calls=num_par_calls)

    # Augment / resize / crop the images
    img_dataset = img_dataset.map(augment_img, num_parallel_calls=num_par_calls)
    img_dataset = img_dataset.batch(batch_size)
    img_dataset = img_dataset.prefetch(4*batch_size)

    image_batch = img_dataset.make_one_shot_iterator().get_next()
    image_batch.set_shape((batch_size, patch_size, patch_size, 3))

    all_depths = tf.convert_to_tensor([1/2, 1/1.5, 1/1, 1/0.5, 1000], tf.float32)
    depth_bins = []
    for i in range(num_depths):
        depth_idx = tf.multinomial(tf.log([5*[1/5]]), num_samples=1)
        depth_bins.append(all_depths[tf.cast(depth_idx[0][0], tf.int32)])

    # Create a staircase depth map that places each image at a number of different depths.
    depth_map = np.concatenate([np.ones((patch_size//num_depths, patch_size)) * i for i in range(num_depths)], axis=0)[:,:,None]
    depth_map = np.tile(depth_map, [batch_size, 1, 1, 1])

    tf.summary.image("input_img", image_batch)
    tf.summary.image("depth_map", depth_map)
    tf.summary.scalar("input_img_max", tf.reduce_max(image_batch))
    tf.summary.scalar("input_img_min", tf.reduce_min(image_batch))
    tf.summary.histogram('depth_bins', depth_bins)

    return image_batch, depth_map, depth_bins
