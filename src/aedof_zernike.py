'''Optimizes an extended-depth-of-field lens using a zernike basis representation. See paper section 4.
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, required=True, help='Path to the training images.')
parser.add_argument('--log_dir', type=str, required=True,
                    help='Directory that checkpoints and tensorboard logfiles will be written to.')
opt = parser.parse_args()

import model
import layers.optics as optics
import layers.deconv as deconv
import edof_reader

import numpy as np
import tensorflow as tf

import os


class ExtendedDepthOfFieldModel(model.Model):
    def __init__(self,
                 sensor_distance,
                 refractive_idcs,
                 wave_lengths,
                 patch_size,
                 ckpt_path,
                 sampling_interval,
                 wave_resolution):
        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.input_sample_interval = sampling_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs

        super(ExtendedDepthOfFieldModel, self).__init__(name='ExtendedDepthOfField', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, zernike_volume, init_gamma, height_map_noise):
        input_img, depth_map = x_train

        with tf.device('/device:GPU:0'):
            # This depth determines the PSF that will be used for deconvolution. It is also optimized.
            target_depth_initializer = tf.constant_initializer(1.)
            target_depth = tf.get_variable(name="target_depth",
                                           shape=(),
                                           dtype=tf.float32,
                                           trainable=True,
                                           initializer=target_depth_initializer)
            target_depth = tf.square(target_depth) # Enforce that depth is positive.
            tf.summary.scalar('target_depth', target_depth)

            optical_system = optics.ZernikeSystem(zernike_volume=zernike_volume,
                                                  target_distance=target_depth,
                                                  wave_resolution=self.wave_resolution,
                                                  upsample=False,
                                                  wave_lengths=self.wave_lengths,
                                                  sensor_resolution=(self.patch_size, self.patch_size),
                                                  height_tolerance=height_map_noise,
                                                  refractive_idcs=self.refractive_idcs,
                                                  input_sample_interval=self.input_sample_interval,
                                                  sensor_distance=self.sensor_distance,
                                                  depth_bins=self.depth_bins)

            # We want to be robust to the noise level. Thus we pick a noise level at random.
            noise_sigma = tf.random_uniform(minval=0.001, maxval=0.02, shape=[])
            sensor_img = optical_system.get_sensor_img(input_img=input_img,
                                                       noise_sigma=noise_sigma,
                                                       depth_dependent=True,
                                                       depth_map=depth_map)
            output_image = tf.cast(sensor_img, tf.float32)

            # Now deconvolve
            pad_width = output_image.shape.as_list()[1] // 2

            output_image = tf.pad(output_image, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
            output_image = deconv.inverse_filter(output_image, output_image, optical_system.target_psf,
                                                 init_gamma=init_gamma)
            output_image = output_image[:, pad_width:-pad_width, pad_width:-pad_width, :]

            return output_image

    def _get_data_loss(self, model_output, ground_truth, margin=10):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.reduce_mean(tf.square(model_output - ground_truth)[:, margin:-margin, margin:-margin, :])

        optics.attach_summaries('output_image', model_output[:,margin:-margin,margin:-margin,:],
                                image=True, log_image=False)
        return loss

    def _get_training_queue(self, batch_size):
        image_batch, depth_batch, self.depth_bins = edof_reader.get_edof_training_queue(opt.img_dir,
                                                                                       patch_size=self.patch_size,
                                                                                       batch_size=batch_size,
                                                                                       num_depths=3,
                                                                                       color=True)

        return (image_batch, depth_batch), image_batch


if __name__ == '__main__':
    tf.reset_default_graph()

    distance = 35.5e-3
    refractive_idcs = np.array([1.499, 1.493, 1.488])
    wave_lenghts = np.array([460, 550, 640]) * 1e-9
    ckpt_path = None
    num_steps = 10001
    patch_size = 1356
    sampling_interval = 3.69e-6
    wave_resolution = 1356, 1356

    eof_model = ExtendedDepthOfFieldModel(distance,
                                          refractive_idcs,
                                          wave_lenghts,
                                          patch_size,
                                          ckpt_path,
                                          sampling_interval=sampling_interval,
                                          wave_resolution=wave_resolution)

    # Generating the zernike basis functions is expensive - if possible, only do it once.
    if not os.path.exists('zernike_volume_%d.npy'%patch_size):
        zernike_volume = optics.get_zernike_volume(resolution=patch_size, n_terms=350).astype(np.float32)
        np.save('zernike_volume_%d.npy'%patch_size, zernike_volume)
    else:
        zernike_volume = np.load('zernike_volume_%d.npy' % patch_size)

    zernike_volume_graph = tf.placeholder(tf.float32, [zernike_volume.shape[0], patch_size, patch_size])

    eof_model.fit(model_params={'init_gamma': 1.5, 'height_map_noise': None, 'zernike_volume': zernike_volume_graph},
                  feed_dict={zernike_volume_graph: zernike_volume},
                  opt_type='Adadelta',
                  opt_params={},
                  batch_size=1,
                  starter_learning_rate=1.,
                  num_steps_until_save=500,
                  num_steps_until_summary=200,
                  logdir=opt.log_dir,
                  num_steps=num_steps)
