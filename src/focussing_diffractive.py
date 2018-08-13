import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import model
import layers.optics as optics

import numpy as np
import tensorflow as tf

from glob import glob
import nn_architectures

import data_readers.edof_reader
import cv2

import layers.deconv as deconv

class RGBCollimator(model.Model):
    def __init__(self,
                 aperture_diameter,
                 distance,
                 refractive_idcs,
                 wave_lengths,
                 patch_size,
                 ckpt_path):

        spot_half_angle = np.arctan2(aperture_diameter/2, distance)
        spot_half_angle_sin = np.sin(spot_half_angle)
        aperture_diffraction_limit = np.amax(wave_lengths / (2 * refractive_idcs * spot_half_angle_sin))

        optical_feature_size = 2e-6
        wave_resolution = 2496,2496

        print("\n" + 50*"*")
        print("Wave resolution is %d. Diffraction limit is %.2e. Optical feature size is %.2e"%\
                (wave_resolution[0], aperture_diffraction_limit, optical_feature_size))
        print(50*"*"+"\n")

        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.distance = distance
        self.input_sample_interval = optical_feature_size
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs

        super(RGBCollimator, self).__init__(name='RGBCollimator', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, init_gamma, hm_reg_scale, noise_sigma, height_map_noise, hm_init_type='random_normal'):
        input_img = x_train
        print("build graph", input_img.get_shape())

        with tf.device('/device:GPU:0'):
            def forward_model(input_field):
                field = optics.height_map_element(input_field,
                                                  wave_lengths=self.wave_lengths,
                                                  height_map_regularizer=optics.laplace_l1_regularizer(hm_reg_scale),
                                                  height_map_initializer=None,
                                                  height_tolerance=height_map_noise,
                                                  refractive_idcs=self.refractive_idcs,
                                                  name='height_map_optics')
                field = optics.circular_aperture(field)
                field = optics.propagate_fresnel(field,
                                                 distance=self.distance,
                                                 input_sample_interval=self.input_sample_interval,
                                                 wave_lengths=self.wave_lengths)
                return field

            optical_system = optics.OpticalSystem(forward_model,
                                                  upsample=False,
                                                  wave_resolution=self.wave_resolution,
                                                  wave_lengths=self.wave_lengths,
                                                  sensor_resolution=(self.patch_size,self.patch_size),
                                                  psf_resolution=(self.patch_size, self.patch_size), # Equals wave resolution
                                                  discretization_size=self.input_sample_interval,
                                                  use_planar_incidence=True)

            if noise_sigma is None:
                noise_sigma = tf.random_uniform(minval=0.001, maxval=0.02, shape=[])

            sensor_img = optical_system.get_sensor_img(input_img=input_img,
                                                       noise_sigma=noise_sigma,
                                                       depth_dependent=False)
            output_image = tf.cast(sensor_img, tf.float32)

            # Now deconvolve
            pad_width = output_image.shape.as_list()[1]//2

            output_image = tf.pad(output_image, [[0,0],[pad_width, pad_width],[pad_width,pad_width],[0,0]])
            output_image = deconv.inverse_filter(output_image, output_image, optical_system.psfs[0], init_gamma=init_gamma)
            output_image = output_image[:,pad_width:-pad_width,pad_width:-pad_width,:]

            optics.attach_summaries('output_image', output_image, image=True, log_image=False)
            return output_image

    def _get_data_loss(self, model_output, ground_truth):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.reduce_mean(tf.square(model_output - ground_truth))
        tf.summary.image('diff_image', tf.square(model_output-ground_truth))
        return loss

    def _get_training_queue(self, batch_size, num_threads=4):
        image_batch, _, _ = data_readers.edof_reader.get_jpg_training_queue('./test_imgs/high_res_images',
                                                                             patch_size=self.patch_size,
                                                                             batch_size=batch_size,
                                                                             color=True)
        return image_batch, image_batch

    def _get_inference_queue(self, img_dir):
        image_batch, _, _ = data_readers.edof_reader.get_jpg_training_queue(img_dir,
                                                                            patch_size=self.patch_size,
                                                                            batch_size=1,
                                                                            color=True,
                                                                            loop=False,
                                                                            filetype='png')
        return image_batch, image_batch


if __name__=='__main__':
    tf.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    aperture_diameter = 5e-3
    distance = 25e-3
    refractive_idcs = np.array([1.4648, 1.4599, 1.4568])
    wave_lengths = np.array([460, 550, 640]) * 1e-9
    ckpt_path = None
    num_steps = 10001
    patch_size = 1248

    rgb_collim_model = RGBCollimator(aperture_diameter,
                                     distance,
                                     refractive_idcs=refractive_idcs,
                                     wave_lengths=wave_lenghts,
                                     patch_size=patch_size,
                                     ckpt_path=ckpt_path)

    rgb_collim_model.fit(model_params = {'hm_reg_scale':1000.0, 'init_gamma':2., 'noise_sigma':None, 'height_map_noise':20e-9},
                        opt_type = 'sgd_with_momentum',
                        opt_params = {'momentum':0.5, 'use_nesterov':True},
                        decay_type = 'polynomial',
                        decay_params = {'decay_steps':num_steps, 'end_learning_rate':1e-9},
                        batch_size=1,
                        starter_learning_rate = 5e-1,
                        num_steps_until_save=500,
                        num_steps_until_summary=100,
                        logdir = '/media/data/checkpoints/flatcam/naive_cases/rgb_collimator_diff_plate_no_difflimit',
                        num_steps = num_steps)
