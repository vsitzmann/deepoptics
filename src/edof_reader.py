import numpy as np
import tensorflow as tf
import os

from glob import glob


def get_edof_training_queue(target_dir, patch_size, batch_size, num_depths=4, color=False,
                            num_threads=4, loop=True, filetype='jpg'):
    if filetype == 'jpg':
        file_list = tf.matching_files(os.path.join(target_dir, '*.jpg'))
    elif filetype == 'png':
        file_list = tf.matching_files(os.path.join(target_dir, '*.png'))

    filename_queue = tf.train.string_input_producer(file_list,
                                                    num_epochs=None if loop else 1,
                                                    shuffle=True if loop else False)

    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)
    if filetype == 'jpg':
        if color:
            print("Using color images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_jpeg(image_file,
                                         channels=1)
    elif filetype == 'png':
        if color:
            print("Using color images")
            image = tf.image.decode_png(image_file,
                                        channels=0)
        else:
            print("Using black and white images")
            image = tf.image.decode_png(image_file,
                                        channels=1)

    image = tf.cast(image, tf.float32)  # Shape [height, width, 1]
    image = tf.expand_dims(image, 0)
    image /= 255.

    # Get the ratio of the patch size to the smallest side of the image
    img_height_width = tf.cast(tf.shape(image)[1:3], tf.float32)

    size_ratio = patch_size / tf.reduce_min(img_height_width)

    # Extract a glimpse from the image
    offset_center = tf.random_uniform([1, 2], minval=0.0 + size_ratio / 2, maxval=1.0 - size_ratio / 2,
                                      dtype=tf.float32)
    offset_center = offset_center * img_height_width

    image = tf.image.extract_glimpse(image, size=[patch_size, patch_size], offsets=offset_center, centered=False,
                                     normalized=False)
    image = tf.squeeze(image, 0)

    all_depths = tf.convert_to_tensor([1 / 2, 1 / 1.5, 1 / 1, 1 / 0.5, 1000], tf.float32)

    depth_bins = []
    for i in range(num_depths):
        depth_idx = tf.multinomial(tf.log([5 * [1 / 5]]), num_samples=1)
        depth_bins.append(all_depths[tf.cast(depth_idx[0][0], tf.int32)])

    test_depth = np.concatenate(
        [np.ones((patch_size // len(depth_bins), patch_size)) * i for i in range(len(depth_bins))], axis=0)[:, :, None]

    if color:
        patch_dims = [patch_size, patch_size, 3]
    else:
        patch_dims = [patch_size, patch_size, 1]

    image_batch, depth_batch = tf.train.batch([image, test_depth],
                                              shapes=[patch_dims, [patch_size, patch_size, 1]],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=4 * batch_size)
    tf.summary.image("input_img", image_batch)
    tf.summary.scalar("input_img_max", tf.reduce_max(image_batch))
    tf.summary.scalar("input_img_min", tf.reduce_min(image_batch))
    tf.summary.histogram('depth', depth_bins)
    tf.summary.image('depth', tf.cast(depth_batch, tf.float32))

    return image_batch, depth_batch, depth_bins
