import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import model
import layers.optics as optics
import layers.deconv as deconv
import edof_reader

import numpy as np
import tensorflow as tf

from glob import glob

class ExtendedDepthOfFieldModel(model.Model):
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
        optical_feature_size = np.amin(wave_lengths / (2*spot_half_angle_sin))

        wave_resolution = 1356, 1356
        optical_feature_size = 5.e-3/wave_resolution[0]

        print("\n" + 50*"*")
        print("Wave resolution is %d. Wave lengths are %s. Diffraction limit is %.2e. Optical feature size is %.2e"%\
                (wave_resolution[0], wave_lengths, aperture_diffraction_limit, optical_feature_size))
        print(50*"*"+"\n")

        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.distance = distance
        self.input_sample_interval = optical_feature_size
        self.patch_size = patch_size
        self.refractive_idcs = refractive_idcs

        super(ExtendedDepthOfFieldModel, self).__init__(name='ExtendedDepthOfField', ckpt_path=ckpt_path)

    def _build_graph(self, x_train, global_step, zernike_volume, hm_reg_scale, init_gamma, height_map_noise, learned_target_depth, hm_init_type='random_normal'):
        input_img, depth_map = x_train

        with tf.device('/device:GPU:0'):
            target_depth_initializer = tf.constant_initializer(1.)
            target_depth = tf.get_variable(name="target_depth",
                                    shape=(),
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=target_depth_initializer)
            target_depth = tf.square(target_depth)
            tf.summary.scalar('target_depth', target_depth)

            optical_system = optics.ZernikeSystem(zernike_volume=zernike_volume,
                                                  target_distance=target_depth,
                                                  wave_resolution=self.wave_resolution,
                                                  upsample=False,
                                                  wave_lengths=self.wave_lengths,
                                                  sensor_resolution=(self.patch_size, self.patch_size),
                                                  psf_resolution=(self.patch_size, self.patch_size),
                                                  height_tolerance=height_map_noise,
                                                  refractive_idcs=self.refractive_idcs,
                                                  input_sample_interval=self.input_sample_interval,
                                                  sensor_distance=self.distance,
                                                  depth_bins=self.depth_bins,
                                                  use_planar_incidence=False)

            noise_sigma = tf.random_uniform(minval=0.001, maxval=0.02, shape=[])
            sensor_img = optical_system.get_sensor_img(input_img=input_img,
                                                       noise_sigma=noise_sigma,
                                                       depth_dependent=True,
                                                       depth_map=depth_map)
            output_image = tf.cast(sensor_img, tf.float32)

            # Now we deconvolve
            pad_width = output_image.shape.as_list()[1]//2

            output_image = tf.pad(output_image, [[0,0],[pad_width, pad_width],[pad_width,pad_width],[0,0]], mode='SYMMETRIC')
            output_image = deconv.inverse_filter(output_image, output_image, optical_system.target_psf, init_gamma=init_gamma)
            output_image = output_image[:,pad_width:-pad_width,pad_width:-pad_width,:]

            optics.attach_summaries('output_image', output_image, image=True, log_image=False)

            return output_image

    def _get_data_loss(self, model_output, ground_truth, margin=50):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.reduce_mean(tf.square(model_output - ground_truth)[:,margin:-margin,margin:-margin,:])
        return loss

    def _get_training_queue(self, batch_size):
        image_batch, depth_batch, self.depth_bins = edof_reader.get_edof_training_data('./test_imgs/high_res_images/',
                                                                                       patch_size=self.patch_size,
                                                                                       batch_size=batch_size,
                                                                                       log_depth_sampling=True,
                                                                                       num_depths=3)
        return (image_batch, depth_batch), image_batch

if __name__=='__main__':
    tf.reset_default_graph()

    aperture_diameter = 5e-3
    distance = 35.5e-3
    refractive_idcs = np.array([1.499, 1.493, 1.488])
    wave_lenghts = np.array([460, 550, 640]) * 1e-9
    ckpt_path = None
    num_steps = 20001
    patch_size = 1356

    eof_model = ExtendedDepthOfFieldModel(aperture_diameter,
                                                distance,
                                                refractive_idcs,
                                                wave_lenghts,
                                                patch_size,
                                                ckpt_path)

    zernike_volume = optics.get_zernike_volume(resolution=patch_size, n_terms=350).astype(np.float32)
    zernike_volume_graph = tf.placeholder(tf.float32, [zernike_volume.shape[0], zernike_volume.shape[1], zernike_volume.shape[1]])

    eof_model.fit(model_params = {'hm_reg_scale':0., 'init_gamma':1.5, 'learned_target_depth':True, 'height_map_noise':None, 'zernike_volume':zernike_volume_graph},
                  feed_dict={zernike_volume_graph:zernike_volume},
                  opt_type = 'Adadelta',
                  opt_params = {},
                  batch_size=1,
                  starter_learning_rate=1.,
                  num_steps_until_save=500,
                  num_steps_until_summary=200,
                  logdir = '/media/data/checkpoints/deepoptics/reproduction/aedof_zernike',
                  num_steps = num_steps)
