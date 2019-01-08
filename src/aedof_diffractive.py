'''Optimizes a diffractive extended-depth-of-field lens. See paper section 4.
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

class ExtendedDepthOfFieldModel(model.Model):
    def __init__(self,
                 sensor_distance,
                 refractive_idcs,
                 wave_lengths,
                 sampling_interval,
                 wave_resolution,
                 patch_size,
                 ckpt_path):

        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sampling_interval = sampling_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs

        super(ExtendedDepthOfFieldModel, self).__init__(name='ExtendedDepthOfField', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, hm_reg_scale, init_gamma, height_map_noise):
        input_img, depth_map = x_train

        with tf.device('/device:GPU:0'):
            height_map = optics.get_fourier_height_map(self.wave_resolution[0],
                                                       0.75,
                                                       height_map_regularizer=optics.laplace_l1_regularizer(
                                                           hm_reg_scale))

            target_depth_initializer = tf.constant_initializer(1.)
            target_depth = tf.get_variable(name="target_depth",
                                           shape=(),
                                           dtype=tf.float32,
                                           trainable=True,
                                           initializer=target_depth_initializer)
            target_depth = tf.square(target_depth)
            tf.summary.scalar('target_depth', target_depth)

            optical_system = optics.SingleLensSetup(height_map=height_map,
                                                    wave_resolution=self.wave_resolution,
                                                    wave_lengths=self.wave_lengths,
                                                    sensor_distance=self.sensor_distance,
                                                    sensor_resolution=(self.patch_size, self.patch_size),
                                                    input_sample_interval=self.sampling_interval,
                                                    refractive_idcs=self.refractive_idcs,
                                                    height_tolerance=height_map_noise,
                                                    use_planar_incidence=False,
                                                    depth_bins=self.depth_bins,
                                                    upsample=False,
                                                    psf_resolution=self.wave_resolution,
                                                    target_distance=target_depth)

            noise_sigma = tf.random_uniform(minval=0.001, maxval=0.01, shape=[])
            sensor_img = optical_system.get_sensor_img(input_img=input_img,
                                                       noise_sigma=noise_sigma,
                                                       depth_dependent=True,
                                                       depth_map=depth_map)
            output_image = tf.cast(sensor_img, tf.float32)

            # Now we deconvolve
            pad_width = output_image.shape.as_list()[1] // 2

            output_image = tf.pad(output_image, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
            output_image = deconv.inverse_filter(output_image, output_image, optical_system.target_psf,
                                                 init_gamma=init_gamma)
            output_image = output_image[:, pad_width:-pad_width, pad_width:-pad_width, :]

            optics.attach_summaries('output_image', output_image, image=True, log_image=False)

            return output_image

    def _get_data_loss(self, model_output, ground_truth, margin=10):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.reduce_mean(tf.square(model_output - ground_truth)[:, margin:-margin, margin:-margin, :])
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

    aperture_diameter = 5e-3
    sensor_distance = 35.5e-3
    refractive_idcs = np.array([1.4648, 1.4599, 1.4568])
    wave_lenghts = np.array([460, 550, 640]) * 1e-9
    ckpt_path = None
    num_steps = 20001
    patch_size = 1248
    sampling_interval = 2e-6
    wave_resolution = 2496, 2496

    eof_model = ExtendedDepthOfFieldModel(sensor_distance=sensor_distance,
                                          refractive_idcs=refractive_idcs,
                                          wave_lengths=wave_lenghts,
                                          patch_size=patch_size,
                                          ckpt_path=ckpt_path,
                                          sampling_interval=sampling_interval,
                                          wave_resolution=wave_resolution)

    eof_model.fit(model_params={'hm_reg_scale': 0., 'init_gamma': 2., 'height_map_noise': 20e-9},
                  opt_type='sgd_with_momentum',
                  opt_params={'momentum': 0.5, 'use_nesterov': True},
                  decay_type='polynomial',
                  decay_params={'decay_steps': num_steps, 'end_learning_rate': 1e-10},
                  batch_size=1,
                  starter_learning_rate=1e-1,
                  num_steps_until_save=500,
                  num_steps_until_summary=200,
                  logdir=opt.log_dir,
                  num_steps=num_steps)
