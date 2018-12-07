'''Optimizes a simple diffractive focusing lens. See paper section 3.2, last paragraph.
'''

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', type=str, required=True, help='Path to the training images.')
parser.add_argument('--log_dir', type=str, required=True, help='Directory that checkpoints and tensorboard logfiles will'
                                                               'be written to.')
opt = parser.parse_args()

import model
import layers.optics as optics

import numpy as np
import tensorflow as tf

import edof_reader

class RGBCollimator(model.Model):
    def __init__(self,
                 sensor_distance,
                 refractive_idcs,
                 wave_lengths,
                 patch_size,
                 sample_interval,
                 wave_resolution,
                 ckpt_path):

        self.wave_res = wave_resolution
        self.wave_lengths = wave_lengths
        self.sensor_distance = sensor_distance
        self.sample_interval = sample_interval
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs

        super(RGBCollimator, self).__init__(name='RGBCollimator', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, global_step, hm_reg_scale, height_map_noise):
        '''
        Builds the graph for this model.

        :param x_train (graph node): input image.
        :param global_step: Global step variable (unused in this model)
        :param hm_reg_scale: Regularization coefficient for laplace l1 regularizer of phaseplate
        :param height_map_noise: Noise added to height map to account for manufacturing deficiencies
        :return: graph node for output image
        '''
        input_img = x_train

        with tf.device('/device:GPU:0'):
            # Input field is a planar wave.
            input_field = tf.ones((1, self.wave_res[0], self.wave_res[1], len(self.wave_lengths)))

            # Planar wave hits aperture: phase is shifted by phaseplate
            field = optics.height_map_element(input_field,
                                              wave_lengths=self.wave_lengths,
                                              height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
                                              height_map_initializer=None,
                                              height_tolerance=height_map_noise,
                                              refractive_idcs=self.refractive_idcs,
                                              name='height_map_optics')
            field = optics.circular_aperture(field)

            # Propagate field from aperture to sensor
            field = optics.propagate_fresnel(field,
                                             distance=self.sensor_distance,
                                             sampling_interval=self.sample_interval,
                                             wave_lengths=self.wave_lengths)

            # The psf is the intensities of the propagated field.
            psfs = optics.get_intensities(field)

            # Downsample psf to image resolution & normalize to sum to 1
            psfs = optics.area_downsampling_tf(psfs, self.patch_size)
            psfs = tf.div(psfs, tf.reduce_sum(psfs, axis=[1,2], keepdims=True))
            optics.attach_summaries('PSF', psfs, image=True, log_image=True)

            # Image formation: PSF is convolved with input image
            psfs = tf.transpose(psfs, [1,2,0,3])
            output_image = optics.img_psf_conv(input_img, psfs)
            output_image = tf.cast(output_image, tf.float32)
            optics.attach_summaries('output_image', output_image, image=True, log_image=False)

            output_image += tf.random_uniform(minval=0.001, maxval=0.02, shape=[])

            return output_image

    def _get_data_loss(self, model_output, ground_truth):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)

        loss = tf.reduce_mean(tf.square(model_output - ground_truth))
        return loss

    def _get_training_queue(self, batch_size):
        image_batch, _, _ = edof_reader.get_edof_training_data(opt.img_dir,
                                                               patch_size=self.patch_size,
                                                               batch_size=batch_size)
        return image_batch, image_batch


if __name__=='__main__':
    tf.reset_default_graph()

    aperture_diameter = 5e-3
    sensor_distance = 25e-3 # Distance of sensor to aperture
    refractive_idcs = np.array([1.4648, 1.4599, 1.4568]) # Refractive idcs of the phaseplate
    wave_lengths = np.array([460, 550, 640]) * 1e-9 # Wave lengths to be modeled and optimized for
    ckpt_path = None
    num_steps = 10001 # Number of SGD steps
    patch_size = 1248 # Size of patches to be extracted from images, and resolution of simulated sensor
    sample_interval = 2e-6 # Sampling interval (size of one "pixel" in the simulated wavefront)
    wave_resolution = 2496, 2496 # Resolution of the simulated wavefront

    rgb_collim_model = RGBCollimator(sensor_distance,
                                     refractive_idcs=refractive_idcs,
                                     wave_lengths=wave_lengths,
                                     patch_size=patch_size,
                                     sample_interval=sample_interval,
                                     wave_resolution=wave_resolution,
                                     ckpt_path=ckpt_path)

    rgb_collim_model.fit(model_params = {'hm_reg_scale':1000.0, 'height_map_noise':20e-9},
                         opt_type = 'sgd_with_momentum',
                         opt_params = {'momentum':0.5, 'use_nesterov':True},
                         decay_type = 'polynomial',
                         decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                         batch_size=1,
                         starter_learning_rate = 5e-1,
                         num_steps_until_save=500,
                         num_steps_until_summary=100,
                         logdir = opt.log_dir,
                         num_steps = num_steps)

