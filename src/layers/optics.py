import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import ifftshift
import fractions


##############################
# Helper functions
##############################
def fspecial(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d


def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def laplacian_filter_tf(img_batch):
    """Laplacian filter. Also considers diagonals.
    """
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])

    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(filter_input, filter=laplacian_filter, padding="SAME")
    return filtered_batch


def laplace_l1_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l1(a_tensor):
        with tf.name_scope('laplace_l1_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
            return scale * tf.reduce_mean(tf.abs(laplace_filtered))

    return laplace_l1


def binary_regularizer(scale):
    if scale == 0.:
        print("Scale of zero disables the regularizer.")

    def binary_reg(a_tensor):
        with tf.name_scope('binary_regularizer'):
            local_losses = tf.square(tf.square(a_tensor) - a_tensor)
            local_losses = tf.cast(local_losses, tf.float32)
            attach_summaries("Local_binary_reg_losses", local_losses, image=True, log_image=True)
            return scale * tf.reduce_mean(local_losses)

    return binary_reg


def laplace_l2_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l2(a_tensor):
        with tf.name_scope('laplace_l2_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
            return scale * tf.reduce_mean(tf.square(laplace_filtered))

    return laplace_l2


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs.reshape([1, 1, 1, -1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts


def get_one_phase_shift_thickness(wave_length, refractive_index):
    '''Calculate the thickness (in meter) of the material necessary to shift incoming light by 2pi.

    :param wave_length: The wave length of the light
    :param refractive_index: The refractive index of the material
    :return: Thickness in meter.
    '''
    # refractive index difference
    delta_N = refractive_index - 1.

    # wave number
    wave_nos = 2. * np.pi / wave_length

    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_N)
    return two_pi_thickness


def attach_summaries(name, var, image=False, log_image=False):
    '''Attaches summaries to var.

    :param name: Name of the variable in tensorboard.
    :param var: Variable to attach summary to.
    :param image: Whether an image summary should be created for var.
    :param log_image: Whether a log image summary should be created for var.
    :return:
    '''

    tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_max', tf.reduce_max(var))

    if image:
        tf.summary.image(name, var, max_outputs=3)
    if log_image and image:
        tf.summary.image(name + '_log', tf.log(var + 1e-12), max_outputs=3)

        var_min = tf.reduce_min(var, axis=(1, 2), keepdims=True)
        var_max = tf.reduce_max(var, axis=(1, 2), keepdims=True)
        var_norm = (var - var_min) / (var_max - var_min)
        tf.summary.image(name + '_norm', var_norm, max_outputs=3)


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2, 3, 0, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def img_psf_conv(img, psf, otf=None, adjoint=False):
    '''Convolves an input image with the point spread function, in the fourier domain.

    :param img: Graph node for input image. Shape (batch_size, height, width, num_channels)
    :param psf: Graph node for psfs. Shape (height, width, batch_size, num_channels)
    :param otf: If available, can pass in otf instead
    :param adjoint: If adjoint convolution should be performed.
    :param circular: If circular convolution should be performed.
    :return: graph node for the result of the convolution.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    psf = tf.convert_to_tensor(psf, dtype=tf.float32)

    img_shape = img.shape.as_list()

    target_side_length = 2 * img_shape[1]

    height_pad = (target_side_length - img_shape[1]) / 2
    width_pad = (target_side_length - img_shape[1]) / 2

    pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
    pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

    img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
    img_shape = img.shape.as_list()

    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)

    result = tf.cast(tf.real(result), tf.float32)

    result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result


def depth_dep_convolution(img, psfs, disc_depth_map):
    """Performs a depth-dependent convolution of an image (with depth map disc_depth_map) and a psf.

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psfs: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        disc_depth_map: The discretized depth map.
        use_fft: Use img_fft_conv or normal conv2d
    """
    # TODO: only convolve with PSFS that are necessary.
    img = tf.cast(img, dtype=tf.float32)
    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])

    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(depth_idx, tf.int16))
        blurred_img = img_psf_conv(img, psf)
        blurred_imgs.append(tf.where(condition,
                                     blurred_img,
                                     zeros_tensor))

    result = tf.reduce_sum(blurred_imgs, axis=0)
    return result


def get_spherical_wavefront_phase(resolution,
                                  physical_size,
                                  wave_lengths,
                                  source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)

    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2].astype(np.float64)

    x = x / N * physical_size
    y = y / M * physical_size

    # Assume distance to source is approx. constant over wave
    curvature = tf.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
    wave_nos = 2. * np.pi / wave_lengths

    phase_shifts = compl_exp_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool(input_image,
                                    [1, factor, factor, 1],
                                    strides=[1, factor, factor, 1],
                                    padding="VALID")
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize_nearest_neighbor(input_image,
                                                         size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool(img_upsampled,
                                    [1, upsample_factor, upsample_factor, 1],
                                    strides=[1, upsample_factor, upsample_factor, 1],
                                    padding="VALID")

    return output_img


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')


##################################
# Optical elements & Propagation
##################################

class Propagation():
    def __init__(self,
                 prop_kernel):
        self.prop_kernel = prop_kernel

    def __call__(self, input_field):
        _, N_orig, _, _ = input_field.shape.as_list()

        # zero padding.
        Npad = N_orig//4 # We would like to pad more, but doesn't fit on GPU

        padded_input_field = tf.pad(input_field,
                                    [[0,0],[Npad,Npad],[Npad,Npad],[0,0]])

        objFT = transp_fft2d( padded_input_field )
        out_field = transp_ifft2d(objFT * self.prop_kernel)

        return out_field[:,Npad:-Npad,Npad:-Npad,:]


class FresnelPropagation(Propagation):
    def __init__(self,
                 wave_resolution,
                 wave_lengths,
                 sampling_interval,
                 propagation_distance):
        N_orig,_ = wave_resolution
        N = int(1.5 * N_orig)

        kernel_shape = (1, N, N, len(wave_lengths))

        [x,y] = np.mgrid[-N//2:N//2,
                         -N//2:N//2]

        # Spatial frequency
        fx = x / (sampling_interval*N) # max frequency = 1/(2*pixel_size)
        fy = y / (sampling_interval*N)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None,:,:,None]
        fy = fy[None,:,:,None]

        squared_sum = np.square(fx) + np.square(fy)
        wls = wave_lengths[None,None,None,:]

        # We create a non-trainable variable so that this computation can be reused
        # from call to call.
        if tf.contrib.framework.is_tensor(propagation_distance):
            tmp = np.float64(wls*np.pi*-1.*squared_sum)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                     initializer=constant_exp_part_init,
                                                     shape=kernel_shape,
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf( propagation_distance * constant_exponent_part, dtype=tf.complex64,
                              name='fresnel_kernel')
        else: # Save some memory
            tmp = np.float64(wls*np.pi*-1.*squared_sum * propagation_distance)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                     initializer=constant_exp_part_init,
                                                     shape=kernel_shape,
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf( constant_exponent_part, dtype=tf.complex64,
                              name='fresnel_kernel')
        super().__init__(prop_kernel=H)


class PhasePlate():
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None):
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        self._build()

    def _build(self):
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += tf.random_uniform(shape=self.height_map.shape,
                                                 minval=-self.height_tolerance,
                                                 maxval=self.height_tolerance,
                                                 dtype=self.height_map.dtype)
            print("Phase plate with manufacturing tolerance %0.2e" % self.height_tolerance)

        self.phase_shifts = phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_idcs)

    def __call__(self, input_field):
        input_field = tf.cast(input_field, tf.complex64)
        return tf.multiply(input_field, self.phase_shifts, name='phase_plate_shift')


def propagate_exact(input_field,
                    distance,
                    input_sample_interval,
                    wave_lengths):
    _, M_orig, N_orig, _ = input_field.shape.as_list()
    # zero padding.
    Mpad = M_orig // 2
    Npad = N_orig // 2
    M = M_orig + 2 * Mpad
    N = N_orig + 2 * Npad
    padded_input_field = tf.pad(input_field,
                                [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2]

    # Spatial frequency
    fx = x / (input_sample_interval * N)  # max frequency = 1/(2*pixel_size)
    fy = y / (input_sample_interval * M)

    # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    fx = fx[None, :, :, None]
    fy = fy[None, :, :, None]

    squared_sum = np.square(fx) + np.square(fy)

    # We create a non-trainable variable so that this computation can be reused
    # from call to call.
    if tf.contrib.framework.is_tensor(distance):
        tmp = np.float64(
            2 * np.pi * (1 / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(distance * constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')
    else:  # Save some memory
        tmp = np.float64(
            2 * np.pi * (distance / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')

    objFT = transp_fft2d(padded_input_field)
    out_field = transp_ifft2d(objFT * H)

    return out_field[:, Mpad:-Mpad, Npad:-Npad, :]


def propagate_fresnel(input_field,
                      distance,
                      sampling_interval,
                      wave_lengths):
    wave_resolution = input_field.shape.as_list()[1:3]
    propagation = FresnelPropagation(wave_resolution=wave_resolution,
                                     propagation_distance=distance,
                                     sampling_interval=sampling_interval,
                                     wave_lengths=wave_lengths)
    return propagation(input_field)


def circular_aperture(input_field):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
             -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float64)
    return aperture * input_field


def height_map_element(input_field,
                       name,
                       wave_lengths,
                       refractive_idcs,
                       block_size=1,
                       height_map_initializer=None,
                       height_map_regularizer=None,
                       height_tolerance=None,  # Default height tolerance is 2 nm.
                       ):
    '''Performs pointwise phaseshifts on input field according to a
    learnable height-map parmeterization of a phaseplate.

    :param input_field: Tensorflow node of input wavefront.
    :param name: Name of this operation.
    :param wave_lengths: Wave lengths to be simulated.
    :param refractive_idcs: Refractive idcs of material at wave_lengths
    :param block_size: Whether each pixel is trainable (block_size=1) or each nxn block has identical thickness (block_size=n)
    :param height_map_initializer: initializer for the height map. If none, initialize at constant thickness of 1e-4 m.
    :param height_map_regularizer: Regularizer for height map.
    :param height_tolerance: Manufacturing tolerance - uniform noise in this range will be added to increase robustness.
    :return: phase-shifted input field.
    '''
    _, height, width, _ = input_field.shape.as_list()
    height_map_shape = [1, height // block_size, width // block_size, 1]

    if height_map_initializer is None:
        init_height_map_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
        height_map_initializer = tf.constant_initializer(init_height_map_value)

    with tf.variable_scope(name, reuse=False):
        height_map_var = tf.get_variable(name="height_map_sqrt",
                                         shape=height_map_shape,
                                         dtype=tf.float64,
                                         trainable=True,
                                         initializer=height_map_initializer)

        height_map_full = tf.image.resize_images(height_map_var, height_map_shape[1:3],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        height_map = tf.square(height_map_full, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def plano_convex_lens(input_field,
                      focal_length,
                      wave_lengths,
                      discretization_step,
                      refractive_idcs):
    input_shape = input_field.shape.as_list()
    _, N, M, _ = input_shape

    convex_radius = (refractive_idcs.reshape([1, 1, 1, -1]) - 1.) * focal_length
    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2].astype(np.float64)

    x = x * discretization_step
    y = y * discretization_step
    x = x.reshape([1, N, M, 1])
    y = y.reshape([1, N, M, 1])

    # This approximates the spherical surface with qaudratic-phase surfaces.
    height_map = - (x ** 2 + y ** 2) / 2. * (1. / convex_radius)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs)

    return element(input_field)


def cubic_phase_shifts(input_field):
    input_shape = input_field.shape.as_list()
    _, N, M, _ = input_shape

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2].astype(np.float64)

    x = x / np.array(N // 2, dtype=np.float64)
    y = y / np.array(M // 2, dtype=np.float64)

    phase_shifts = np.mod(20. * np.pi * (x ** 3 + y ** 3), 2. * np.pi)
    phase_shifts = phase_shifts[None, :, :, None]

    phase_shifts = compl_exp_tf(phase_shifts, dtype=tf.complex128)
    input_field = tf.cast(input_field, tf.complex128)

    return input_field * phase_shifts


def fourier_element(input_field,
                    name,
                    wave_lengths,
                    refractive_idcs,
                    frequency_range=0.5,
                    height_map_regularizer=None,
                    height_tolerance=None,  # Default height tolerance is 2 nm.
                    ):
    _, height, width, _ = input_field.shape.as_list()
    fourier_initializer = tf.zeros_initializer()

    with tf.variable_scope(name, reuse=False):
        fourier_vars_real = tf.get_variable('fourier_coeffs_real',
                                            shape=[1, int(height * frequency_range), int(width * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_vars_cplx = tf.get_variable('fourier_coeffs_cmplx',
                                            shape=[1, int(height * frequency_range), int(width * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_coeffs = tf.complex(fourier_vars_real, fourier_vars_cplx)
        attach_summaries("Fourier_coeffs", tf.abs(fourier_coeffs), image=True, log_image=False)
        size = int(height * frequency_range)
        padding_width = (height - size) // 2
        fourier_coeffs_padded = tf.pad(fourier_coeffs,
                                       [[0, 0], [padding_width, padding_width], [padding_width, padding_width], [0, 0]])
        height_map = tf.real(transp_ifft2d(ifftshift2d_tf(fourier_coeffs_padded)))

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def zernike_element(input_field,
                    zernike_volume,
                    name,
                    wave_lengths,
                    refractive_idcs,
                    zernike_initializer=None,
                    height_map_regularizer=None,
                    height_tolerance=None,  # Default height tolerance is 2 nm.
                    zernike_scale=1e5,
                    ):
    _, height, width, _ = input_field.shape.as_list()

    num_zernike_coeffs = zernike_volume.shape.as_list()[0]

    if zernike_initializer is None:
        zernike_initializer = tf.zeros_initializer()  # tf.random_normal_initializer(stddev=1e-6)

    with tf.variable_scope(name, reuse=False):
        zernike_coeffs = tf.get_variable('zernike_coeffs',
                                         shape=[num_zernike_coeffs, 1, 1],
                                         dtype=tf.float32,
                                         trainable=True,
                                         initializer=zernike_initializer)
        # mask = np.ones([num_zernike_coeffs, 1, 1])
        # mask[0] = 0.
        # zernike_coeffs *= mask/zernike_scale

        for i in range(num_zernike_coeffs):
            tf.summary.scalar('zernike_coeff_%d' % i, tf.squeeze(zernike_coeffs[i, :, :]))

        height_map = tf.reduce_sum(zernike_coeffs * zernike_volume, axis=0)
        height_map = tf.expand_dims(tf.expand_dims(height_map, 0), -1, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        height_map_summary = (height_map - tf.reduce_min(height_map)) / (
                    tf.reduce_max(height_map) - tf.reduce_min(height_map))
        attach_summaries("Height_map", height_map_summary, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + tf.random_normal(image.shape, 0.0, stddev, dtype=dtype)


def get_vanilla_height_map(side_length,
                           height_map_regularizer=None,
                           height_map_initializer=None,
                           name='height_map'):
    height_map_shape = [1, side_length, side_length, 1]

    if height_map_initializer is None:
        init_height_map_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
        height_map_initializer = tf.constant_initializer(init_height_map_value)
        print("Using default in initializer")
    else:
        print("Using passed in initializer")

    with tf.variable_scope(name, reuse=False):
        height_map = tf.get_variable(name="height_map",
                                     shape=height_map_shape,
                                     dtype=tf.float64,
                                     trainable=True,
                                     initializer=height_map_initializer)
        # height_map = tf.square(height_map_sqrt, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

        return tf.cast(height_map, tf.float64)


def get_fourier_height_map(side_length,
                           frequency_range=0.5,
                           height_map_regularizer=None,
                           name='fourier_height_map'):
    height_map_shape = [1, side_length, side_length, 1]

    fourier_initializer = tf.zeros_initializer()

    with tf.variable_scope(name, reuse=False):
        fourier_vars_real = tf.get_variable('fourier_coeffs_real',
                                            shape=[1, int(side_length * frequency_range),
                                                   int(side_length * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_vars_cplx = tf.get_variable('fourier_coeffs_cmplx',
                                            shape=[1, int(side_length * frequency_range),
                                                   int(side_length * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_coeffs = tf.complex(fourier_vars_real, fourier_vars_cplx)
        attach_summaries("Fourier_coeffs", tf.abs(fourier_coeffs), image=True, log_image=False)
        padding_width = int((1 - frequency_range) * side_length) // 2
        fourier_coeffs_padded = tf.pad(fourier_coeffs,
                                       [[0, 0], [padding_width, padding_width], [padding_width, padding_width], [0, 0]])
        print(fourier_coeffs_padded.shape.as_list())
        height_map = tf.real(transp_ifft2d(ifftshift2d_tf(fourier_coeffs_padded)))

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)
        return height_map


class SingleLensSetup():
    def __init__(self,
                 height_map,
                 wave_resolution,
                 wave_lengths,
                 sensor_distance,
                 sensor_resolution,
                 input_sample_interval,
                 refractive_idcs,
                 height_tolerance,
                 noise_model=gaussian_noise,
                 psf_resolution=None,
                 target_distance=None,
                 use_planar_incidence=True,
                 upsample=True,
                 depth_bins=None):

        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs

        self.wave_resolution = wave_resolution
        if psf_resolution is None:
            psf_resolution = wave_resolution
        self.psf_resolution = psf_resolution

        self.sensor_distance = sensor_distance
        self.noise_model = noise_model
        self.sensor_resolution = sensor_resolution
        self.input_sample_interval = input_sample_interval

        self.use_planar_incidence = use_planar_incidence
        self.upsample = upsample
        self.target_distance = target_distance
        self.depth_bins = depth_bins

        self.height_tolerance = height_tolerance
        self.height_map = height_map

        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)
        self.pixel_size = self.input_sample_interval * np.array(wave_resolution) / np.array(sensor_resolution)

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_resolution[0]))

        self.optical_element = PhasePlate(wave_lengths=self.wave_lengths,
                                          height_map=self.height_map,
                                          refractive_idcs=self.refractive_idcs,
                                          height_tolerance=self.height_tolerance)
        self.get_psfs()

    def get_psfs(self):
        # Sort the point source distances in increasing order
        if self.use_planar_incidence:
            input_fields = [tf.ones(self.wave_resolution, dtype=tf.float32)[None, :, :, None]]
        else:
            distances = self.depth_bins

            if self.target_distance is not None:
                distances += [self.target_distance]

            N, M = self.wave_resolution
            [x, y] = np.mgrid[-N // 2:N // 2,
                     -M // 2:M // 2].astype(np.float64)

            x = x / N * self.physical_size
            y = y / M * self.physical_size

            squared_sum = x ** 2 + y ** 2

            wave_nos = 2. * np.pi / self.wave_lengths
            wave_nos = wave_nos.reshape([1, 1, 1, -1])

            input_fields = []
            for distance in distances:
                # Assume distance to source is approx. constant over wave
                curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
                curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)

                spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=tf.complex64)
                input_fields.append(spherical_wavefront)

        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for depth_idx, input_field in enumerate(input_fields):
                field = self.optical_element(input_field)
                field = circular_aperture(field)
                sensor_incident_field = propagate_fresnel(field,
                                                          distance=self.sensor_distance,
                                                          sampling_interval=self.input_sample_interval,
                                                          wave_lengths=self.wave_lengths)
                psf = get_intensities(sensor_incident_field)

                if not self.upsample:
                    psf = area_downsampling_tf(psf, self.psf_resolution[0])

                psf = tf.div(psf, tf.reduce_sum(psf, axis=[1, 2], keepdims=True), name='psf_depth_idx_%d' % depth_idx)

                attach_summaries('PSF_depth_idx_%d' % depth_idx, psf, image=True, log_image=True)
                psfs.append(tf.transpose(psf, [1, 2, 0, 3]))  # (Height, width, 1, channels)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            attach_summaries('target_psf', tf.transpose(self.target_psf, [2, 0, 1, 3]), image=True, log_image=True)

        self.psfs = psfs

    def get_sensor_img(self,
                       input_img,
                       noise_sigma=0.001,
                       depth_dependent=False,
                       depth_map=None,
                       otfs=None):
        """"""
        # Upsample input_img to match wave resolution.
        if self.upsample:
            print("Images are upsampled to wave resolution")
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            print("Images are not upsampled to wave resolution")

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            sensor_img = img_psf_conv(input_img, self.psfs[0], otf=otfs)

        # Down sample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        noisy_img = self.noise_model(sensor_img, noise_sigma)

        # print("Additive noise of %0.2e"%noise_sigma)
        attach_summaries("Sensor_img", noisy_img, image=True, log_image=False)

        return noisy_img


class ZernikeSystem():
    def __init__(self,
                 zernike_volume,
                 wave_resolution,
                 wave_lengths,
                 sensor_distance,
                 sensor_resolution,
                 input_sample_interval,
                 refractive_idcs,
                 height_tolerance,
                 noise_model=gaussian_noise,
                 psf_resolution=None,
                 target_distance=None,
                 use_planar_incidence=True,
                 upsample=True,
                 depth_bins=None,
                 zernike_scale=1e8):

        self.sensor_distance = sensor_distance
        self.zernike_volume = zernike_volume
        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.use_planar_incidence = use_planar_incidence
        self.depth_bins = depth_bins
        self.noise_model = noise_model
        self.upsample = upsample
        self.target_distance = target_distance
        self.zernike_volume = zernike_volume
        self.height_tolerance = height_tolerance
        self.input_sample_interval = input_sample_interval
        self.zernike_scale = zernike_scale
        self.refractive_idcs = refractive_idcs
        self.sensor_resolution = sensor_resolution

        if psf_resolution is None:
            psf_resolution = wave_resolution
        self.psf_resolution = psf_resolution

        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_resolution[0]))

        self.get_psfs()

    def _build_height_map(self):
        height_map_shape = [1, self.wave_resolution[0], self.wave_resolution[1], 1]

        num_zernike_coeffs = self.zernike_volume.shape.as_list()[0]

        zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
        zernike_inits[3] = -51.
        zernike_initializer = tf.constant_initializer(zernike_inits)

        self.zernike_coeffs = tf.get_variable('zernike_coeffs',
                                              shape=[num_zernike_coeffs, 1, 1],
                                              dtype=tf.float32,
                                              trainable=True,
                                              initializer=zernike_initializer)
        # mask = np.ones([num_zernike_coeffs, 1, 1])
        # mask[0] = 0.
        # self.zernike_coeffs *= mask

        for i in range(num_zernike_coeffs):
            tf.summary.scalar('zernike_coeff_%d' % i, tf.squeeze(self.zernike_coeffs[i, :, :]))

        self.height_map = tf.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0)
        self.height_map = tf.expand_dims(tf.expand_dims(self.height_map, 0), -1, name='height_map')

        attach_summaries("Height_map", self.height_map, image=True, log_image=False)

        self.element = PhasePlate(wave_lengths=self.wave_lengths,
                                  height_map=self.height_map,
                                  refractive_idcs=self.refractive_idcs,
                                  height_tolerance=self.height_tolerance)

    def get_psfs(self):
        # Sort the point source distances in increasing order
        self._build_height_map()

        if self.use_planar_incidence:
            input_fields = [tf.ones(self.wave_resolution, dtype=tf.float32)[None, :, :, None]]
        else:
            distances = self.depth_bins

            if self.target_distance is not None:
                distances += [self.target_distance]

            N, M = self.wave_resolution
            [x, y] = np.mgrid[-N // 2:N // 2,
                     -M // 2:M // 2].astype(np.float64)

            x = x / N * self.physical_size
            y = y / M * self.physical_size

            squared_sum = x ** 2 + y ** 2

            wave_nos = 2. * np.pi / self.wave_lengths
            wave_nos = wave_nos.reshape([1, 1, 1, -1])

            input_fields = []
            for distance in distances:
                # Assume distance to source is approx. constant over wave
                curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
                curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)

                spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=tf.complex64)
                input_fields.append(spherical_wavefront)

        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for depth_idx, input_field in enumerate(input_fields):
                field = self.element(input_field)
                field = circular_aperture(field)
                sensor_incident_field = propagate_fresnel(field,
                                                          distance=self.sensor_distance,
                                                          sampling_interval=self.input_sample_interval,
                                                          wave_lengths=self.wave_lengths)
                psf = get_intensities(sensor_incident_field)

                if not self.upsample:
                    psf = area_downsampling_tf(psf, self.psf_resolution[0])

                psf = tf.div(psf, tf.reduce_sum(psf, axis=[1, 2], keepdims=True), name='psf_depth_idx_%d' % depth_idx)

                attach_summaries('PSF_depth_idx_%d' % depth_idx, psf, image=True, log_image=True)
                psfs.append(tf.transpose(psf, [1, 2, 0, 3]))  # (Height, width, 1, channels)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            attach_summaries('target_psf', tf.transpose(self.target_psf, [2, 0, 1, 3]), image=True)

        self.psfs = psfs

    def get_sensor_img(self,
                       input_img,
                       noise_sigma=0.001,
                       depth_dependent=False,
                       depth_map=None,
                       otfs=None):
        """"""
        # Upsample input_img to match wave resolution.
        if self.upsample:
            print("Images are upsampled to wave resolution")
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            print("Images are not upsampled to wave resolution")

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            print("%d psfs" % len(self.psfs))
            sensor_img = img_psf_conv(input_img, self.psfs[0])  # , otf=otfs)

        # Down sample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        noisy_img = self.noise_model(sensor_img, noise_sigma)

        # print("Additive noise of %0.2e"%noise_sigma)
        attach_summaries("Sensor_img", noisy_img, image=True, log_image=False)

        return noisy_img


class OpticalSystem():
    def __init__(self,
                 forward_model,
                 wave_resolution,
                 wave_lengths,
                 sensor_resolution,
                 noise_model=gaussian_noise,
                 psf_resolution=None,
                 target_distance=None,
                 discretization_size=1e-6,
                 use_planar_incidence=True,
                 upsample=True,
                 depth_bins=None,
                 zernike_scale=1e5):

        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.use_planar_incidence = use_planar_incidence
        self.depth_bins = depth_bins
        self.discretization_size = discretization_size
        self.sensor_resolution = sensor_resolution
        self.noise_model = noise_model
        self.upsample = upsample
        self.target_distance = target_distance
        self.forward_model = forward_model

        if psf_resolution is None:
            psf_resolution = wave_resolution
        self.psf_resolution = psf_resolution

        self.physical_size = float(self.wave_resolution[0] * self.discretization_size)
        self.pixel_size = discretization_size * np.array(wave_resolution) / np.array(sensor_resolution)

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_resolution[0]))

        self.get_psfs()

    def get_psfs(self):
        # Sort the point source distances in increasing order
        if self.use_planar_incidence:
            input_fields = [tf.ones(self.wave_resolution, dtype=tf.float32)[None, :, :, None]]
        else:
            distances = self.depth_bins

            if self.target_distance is not None:
                distances += [self.target_distance]

            N, M = self.wave_resolution
            [x, y] = np.mgrid[-N // 2:N // 2,
                     -M // 2:M // 2].astype(np.float64)

            x = x / N * self.physical_size
            y = y / M * self.physical_size

            squared_sum = x ** 2 + y ** 2

            wave_nos = 2. * np.pi / self.wave_lengths
            wave_nos = wave_nos.reshape([1, 1, 1, -1])

            input_fields = []
            for distance in distances:
                # Assume distance to source is approx. constant over wave
                curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
                curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)

                spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=tf.complex64)
                input_fields.append(spherical_wavefront)

        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for depth_idx, input_field in enumerate(input_fields):
                sensor_incident_field = self.forward_model(input_field)
                psf = get_intensities(sensor_incident_field)

                ## Blur the PSF
                # gauss_kernel = fspecial(shape=(3,3), sigma=0.5).reshape(3,3,1,1).astype(np.float32)
                # psf = tf.nn.depthwise_conv2d(psf, gauss_kernel, stride=(1,1,1,1), padding='SAME')

                if not self.upsample:
                    psf = area_downsampling_tf(psf, self.psf_resolution[0])

                psf = tf.div(psf, tf.reduce_sum(psf, axis=[1, 2], keepdims=True), name='psf_depth_idx_%d' % depth_idx)

                attach_summaries('PSF_depth_idx_%d' % depth_idx, psf, image=True, log_image=True)
                psfs.append(tf.transpose(psf, [1, 2, 0, 3]))  # (Height, width, 1, channels)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            attach_summaries('target_psf', tf.transpose(self.target_psf, [2, 0, 1, 3]), image=True)

        self.psfs = psfs

    def get_sensor_img(self,
                       input_img,
                       noise_sigma=0.001,
                       depth_dependent=False,
                       depth_map=None,
                       otfs=None):
        """"""
        # Upsample input_img to match wave resolution.
        if self.upsample:
            print("Images are upsampled to wave resolution")
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            print("Images are not upsampled to wave resolution")

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            sensor_img = img_psf_conv(input_img, self.psfs[0], otf=otfs)

        # Down sample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        noisy_img = self.noise_model(sensor_img, noise_sigma)

        # print("Additive noise of %0.2e"%noise_sigma)
        attach_summaries("Sensor_img", noisy_img, image=True, log_image=False)

        return noisy_img
