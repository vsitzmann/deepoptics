import tensorflow as tf
import layers.optics as optics
from collections import namedtuple
import numpy as np

def inverse_filter(blurred, estimate, psf, gamma=None, otf=None, init_gamma=1.5):
     """Implements Weiner deconvolution in the frequency domain, with circular boundary conditions.

     Args:
         blurred: image with shape (batch_size, height, width, num_img_channels)
         estimate: image with shape (batch_size, height, width, num_img_channels)
         psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)

     TODO precompute OTF, adj_filt_img.
     """
     img_shape = blurred.shape.as_list()

     if gamma is None:
        gamma_initializer = tf.constant_initializer(init_gamma)
        gamma = tf.get_variable(name="wiener_gamma",
                                shape=(),
                                dtype=tf.float32,
                                trainable=True,
                                initializer=gamma_initializer)
        gamma = tf.square(gamma)
        tf.summary.scalar('gamma', gamma)

     a_tensor_transp = tf.transpose(blurred, [0,3,1,2])
     estimate_transp = tf.transpose(estimate, [0,3,1,2])
     # Everything has shape (batch_size, num_channels, height, width)
     img_fft = tf.fft2d(tf.complex(a_tensor_transp, 0.))
     if otf is None:
       otf = optics.psf2otf(psf, output_size=img_shape[1:3])
       otf = tf.transpose(otf, [2,3,0,1])

     adj_conv = img_fft * tf.conj(otf)
     numerator = adj_conv + tf.fft2d(tf.complex(gamma*estimate_transp, 0.))

     kernel_mags = tf.square(tf.abs(otf))

     denominator = tf.complex(kernel_mags + gamma, 0.0)
     filtered = tf.div(numerator, denominator)
     cplx_result = tf.ifft2d(filtered)
     real_result = tf.real(cplx_result)
     # Get back to (batch_size, num_channels, height, width)
     result = tf.transpose(real_result, [0,2,3,1])
     return result

