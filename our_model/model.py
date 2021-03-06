import numpy as np
import tensorflow as tf
import time
from IPython import display
import datetime
import os
import logging


class Reconstruct_R1M(tf.keras.layers.Layer):
    def __init__(self, filters, name='RR'):
        super(Reconstruct_R1M, self).__init__(name=name)
        self.num_outputs = filters

    def build(self, input_shape):
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, 1, 1, self.num_outputs), dtype='float32'))

    def call(self, input):
        v = tf.math.add(input, tf.constant(1e-6, dtype=tf.float32))
        vt = tf.transpose(v, perm=[0, 2, 1, 3])
        rank1m = tf.multiply(tf.multiply(v, vt), self.w)
        return rank1m


class Weight_R1M(tf.keras.layers.Layer):
    def __init__(self, name='WR1M'):
        super(Weight_R1M, self).__init__(name=name)

    def build(self, input_shape):
        w_init = tf.keras.initializers.RandomUniform(minval=0, maxval=4.0)
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, 1, 1, input_shape[-1]), dtype='float32'))

    def call(self, input):
        self.w.assign(tf.nn.relu(self.w))
        return tf.multiply(input, self.w)


class Downpixel(tf.keras.layers.Layer):
    def __init__(self, r, name=None):
        super(Downpixel, self).__init__(name=name)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        X = tf.nn.space_to_depth(
            input=I, block_size=r, data_format='NHWC', name=None)
        return X

    def call(self, inputs):
        r = self.r
        kernel = tf.ones(shape=(r, r, 1, 1), dtype=tf.float32)/(r*r)
        conv = tf.nn.conv2d(inputs, kernel, padding='SAME', strides=(1, 1))
        return self._phase_shift(conv)


class Subpixel(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, r, padding='valid', data_format=None,
                 strides=(1, 1), activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Subpixel, self).__init__(filters=r*r*filters, kernel_size=kernel_size,
                                       strides=strides, padding=padding, data_format=data_format,
                                       activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                       kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        X = tf.nn.depth_to_space(
            input=I, block_size=r, data_format='NHWC', name=None)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], unshifted[3]/(self.r*self.r))


class Sum_R1M(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Sum_R1M, self).__init__(name=name)

    def call(self, input):
        return tf.reduce_sum(input, axis=-1, keepdims=True)


class Symmetry_R1M(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Symmetry_R1M, self).__init__(name=name)

    def build(self, input_shape):
        ones = tf.ones(shape=(input_shape[1], input_shape[2]), dtype='float32')
        diag = tf.linalg.band_part(ones, 0, 0)*0.5
        upper = tf.linalg.band_part(ones, 0, -1)

        self.w = upper - diag
        self.w = tf.expand_dims(self.w, 0)
        self.w = tf.expand_dims(self.w, -1)

    def call(self, input):
        up = tf.multiply(input, self.w)
        low = tf.transpose(up, perm=[0, 2, 1, 3])
        return up + low


class Normal(tf.keras.layers.Layer):
    def __init__(self, input_dim, name=None):
        super(Normal, self).__init__(name=name)
        self.dim = input_dim

    def build(self, input_shape):
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, self.dim, 1, 1), dtype='float32'), trainable=True)

    def call(self, inputs):
        rowsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=1, keepdims=True))
        colsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=2, keepdims=True))
        sumele = tf.math.multiply(rowsr, colsr)
        Div = tf.math.divide_no_nan(inputs, sumele)
        self.w.assign(tf.nn.relu(self.w))
        WT = tf.transpose(self.w, perm=[0, 2, 1, 3])
        M = tf.multiply(self.w, WT)
        return tf.multiply(Div, M)


def block_downsample_decomposition(len_size, channels_decompose, input_len_size, input_channels, downsample_ratio, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Downpixel(downsample_ratio))
    result.add(tf.keras.layers.Conv2D(channels_decompose, [1, len_size], strides=(1, 1), padding='valid', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(Weight_R1M())
    result.add(Reconstruct_R1M(channels_decompose))
    return result


def block_rank1channels_convolution(channels, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels, [1, 1], strides=1, padding='same', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      name=name))
    result.add(Weight_R1M())
    result.add(Symmetry_R1M())
    return result


def block_upsample_convolution(channels, input_len_size, input_channels, upsample_ratio, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Subpixel(filters=int(channels), kernel_size=(3, 3), r=upsample_ratio,
                        activation='relu', use_bias=False, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(Symmetry_R1M())
    return result


def block_rank1_estimation(dims, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Sum_R1M())
    result.add(Normal(dims))
    return result


def block_channel_combination(channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(channels, [1, 1], strides=1, padding='same', data_format="channels_last",
                                      kernel_initializer=tf.keras.initializers.RandomUniform(
                                          minval=0., maxval=1./channels),
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      activation='relu', use_bias=False))
    return result


def make_generator_model(len_high_size=128, scale=4):

    len_low_size_x2 = int(len_high_size/(scale/2))
    len_low_size_x4 = int(len_high_size/scale)
    len_low_size_x8 = int(len_high_size/(scale*2))
    inp = tf.keras.layers.Input(
        shape=(len_high_size, len_high_size, 1), name='in', dtype=tf.float32)

    low_x2 = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid', name='p_x2')(inp)
    low_x4 = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid', name='p_x4')(low_x2)

    dsd_x4 = block_downsample_decomposition(len_size=len_low_size_x4, channels_decompose=384, input_len_size=len_high_size,
                                            input_channels=1, downsample_ratio=4, name='dsd_x4')
    rech_x4 = dsd_x4(inp)
    r1c = block_rank1channels_convolution(
        channels=128, input_len_size=len_low_size_x4, input_channels=384, name='r1c_x4')
    sym_x4 = r1c(rech_x4)
    r1e = block_rank1_estimation(
        dims=len_low_size_x4, input_len_size=len_low_size_x4, input_channels=384, name='r1e_x4')
    out_low_x4 = r1e(rech_x4)

    usc_x4 = block_upsample_convolution(
        channels=128, input_len_size=len_low_size_x4, input_channels=128, upsample_ratio=2, name='usc_x4')
    sym_x4 = usc_x4(sym_x4)

    dsd_x2 = block_downsample_decomposition(len_size=len_low_size_x2, channels_decompose=768, input_len_size=len_high_size,
                                            input_channels=1, downsample_ratio=2, name='dsd_x2')
    rech_x2 = dsd_x2(inp)
    r1c_x2 = block_rank1channels_convolution(
        channels=128, input_len_size=len_low_size_x2, input_channels=768, name='r1c_x2')
    sym_x2 = r1c_x2(rech_x2)
    r1e_x2 = block_rank1_estimation(
        dims=len_low_size_x2, input_len_size=len_low_size_x2, input_channels=768, name='r1e_x2')
    out_low_x2 = r1e_x2(rech_x2)

    concat = tf.keras.layers.concatenate([sym_x4, sym_x2], axis=-1)
    concat = block_channel_combination(channels=128, name='cc_x2')(concat)
    usc_x2 = block_upsample_convolution(
        channels=64, input_len_size=len_low_size_x2, input_channels=128, upsample_ratio=2, name='usc_x2')
    sym = usc_x2(concat)

    Sumh = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                  strides=(1, 1), padding='same',
                                  data_format="channels_last",
                                  kernel_constraint=tf.keras.constraints.NonNeg(),
                                  activation='relu', use_bias=False, name='sum_high')(sym)
    high_out = Normal(int(len_high_size), name='out_high')(Sumh)

    model = tf.keras.models.Model(
        inputs=[inp], outputs=[out_low_x2, out_low_x4, high_out, low_x2, low_x4])
    return model


def block_rank1_decompose_reconstruct(len_size, channels_decompose, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels_decompose, [1, len_size], strides=(1, 1), padding='valid', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(Reconstruct_R1M(channels_decompose))
    return result


def block_down_convolution(channels, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format="channels_last",
                                      activation=None, use_bias=False,
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(tf.keras.layers.LeakyReLU(0.2))
    result.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid'))
    return result


def make_discriminator_model(len_high_size=128, scale=4):
    '''PatchGAN 1 pixel of output represents X pixels of input: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
    The "70" is implicit, it's not written anywhere in the code
    but instead emerges as a mathematical consequence of the network architecture.
    The math is here: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    compute input size from a given output size:
    f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize; fix output_size as 1 '''
    len_x1 = int(len_high_size)
    len_x2 = int(len_high_size/(scale/2))
    len_x4 = int(len_high_size/scale)
    len_x8 = int(len_high_size/(scale*2))
    inp = tf.keras.layers.Input(
        shape=(len_high_size, len_high_size, 1), name='in', dtype=tf.float32)

    b_r1dr = block_rank1_decompose_reconstruct(len_size=len_x1,
                                               channels_decompose=512, input_len_size=len_x1,
                                               input_channels=1, name='r1dr_x1')
    r1dr_x1 = b_r1dr(inp)
    b_dc = block_down_convolution(
        channels=80, input_len_size=len_x1, input_channels=512, name='dc_x1')
    dc_x1 = b_dc(r1dr_x1)

    ratio = 2
    dp_x2 = Downpixel(r=ratio, name='dp_x2')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(len_size=len_x2,
                                               channels_decompose=512, input_len_size=len_x2,
                                               input_channels=ratio**2, name='r1dr_x2')
    r1dr_x2 = b_r1dr(dp_x2)
    b_r1c = block_rank1channels_convolution(
        channels=40, input_len_size=len_x2, input_channels=512, name='r1c_x2')
    r1c_x2 = b_r1c(r1dr_x2)

    concat_x1_x2 = tf.keras.layers.Concatenate()([r1c_x2, dc_x1])
    b_dc = block_down_convolution(
        channels=120, input_len_size=len_x2, input_channels=120, name='dc_x2')
    dc_x2 = b_dc(concat_x1_x2)

    ratio = 4
    dp_x4 = Downpixel(r=ratio, name='dp_x4')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(
        len_size=len_x4, channels_decompose=256, input_len_size=len_x4, input_channels=ratio**2, name='r1dr_x4')
    r1dr_x4 = b_r1dr(dp_x4)
    b_r1c = block_rank1channels_convolution(
        channels=20, input_len_size=len_x4, input_channels=256, name='r1c_x4')
    r1c_x4 = b_r1c(r1dr_x4)

    concat_x2_x4 = tf.keras.layers.Concatenate()([r1c_x4, dc_x2])
    b_dc = block_down_convolution(
        channels=60, input_len_size=len_x4, input_channels=140, name='dc_x4')
    dc_x4 = b_dc(concat_x2_x4)

    ratio = 8
    dp_x8 = Downpixel(r=ratio, name='dp_x8')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(
        len_size=len_x8, channels_decompose=128, input_len_size=len_x8, input_channels=ratio**2, name='r1dr_x8')
    r1dr_x8 = b_r1dr(dp_x8)
    b_r1c = block_rank1channels_convolution(
        channels=10, input_len_size=len_x8, input_channels=128, name='r1c_x8')
    r1c_x8 = b_r1c(r1dr_x8)

    concat_x4_x8 = tf.keras.layers.Concatenate()([r1c_x8, dc_x4])
    b_dc = block_down_convolution(
        channels=80, input_len_size=len_x8, input_channels=70, name='dc_x8')
    dc_x8 = b_dc(concat_x4_x8)

    conv = tf.keras.layers.Conv2D(filters=80, kernel_size=(
        1, 1), strides=1, padding='same')(dc_x8)
    conv = tf.keras.layers.Flatten()(conv)
    last = tf.keras.layers.Dense(1, activation=None)(conv)
    return tf.keras.Model(inputs=inp, outputs=last)


def discriminator_bce_loss(real_output, fake_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_bce_loss(d_pred):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(d_pred), d_pred)
    return gan_loss


def generator_ssim_loss(y_pred, y_true):  # , m_filter):
    return tf.reduce_mean((1 - tf.image.ssim(y_pred, y_true, max_val=1.0, filter_size=11))/2.0)


def generator_mse_loss(y_pred, y_true):  # , m_filter):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    diff = mse(y_pred, y_true)
    diff = tf.reduce_sum(diff, [1,2])
    diff = tf.reduce_mean(diff)
    return diff


# @tf.function
# solution: https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-615870307
def _train_step_generator(Gen, Dis, imgl, imgr, loss_filter, loss_weights, opts):
    with tf.GradientTape() as x, tf.GradientTape() as gen_tape_high:
        fake_hic = Gen(imgl, training=True)
        fake_hic_l_x2 = fake_hic[0]
        # imgl_x2 = fake_hic[4]
        imgl_x2 = fake_hic[3]
        mfilter_low = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l_x2 = tf.multiply(fake_hic_l_x2, mfilter_low)
        imgl_x2_filter = tf.multiply(imgl_x2, mfilter_low)

        fake_hic_l_x4 = fake_hic[1]
        # imgl_x4 = fake_hic[5]
        imgl_x4 = fake_hic[4]
        mfilter_low = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l_x4 = tf.multiply(fake_hic_l_x4, mfilter_low)
        imgl_x4_filter = tf.multiply(imgl_x4, mfilter_low)

        loss_low_ssim_x2 = generator_ssim_loss(
            fake_hic_l_x2, imgl_x2_filter)
        loss_low_mse_x2 = generator_mse_loss(fake_hic_l_x2, imgl_x2_filter)

        loss_low_ssim_x4 = generator_ssim_loss(
            fake_hic_l_x4, imgl_x4_filter)
        loss_low_mse_x4 = generator_mse_loss(fake_hic_l_x4, imgl_x4_filter)

        loss_low_ssim = (loss_low_ssim_x4*4.0 + loss_low_ssim_x2*16.0)/20.0
        loss_low_mse = (loss_low_mse_x4*4.0 + loss_low_mse_x2*16.0)/20.0
        loss_low = loss_low_ssim + loss_low_mse

        fake_hic_h = fake_hic[2]
        mfilter_high = tf.expand_dims(loss_filter[2], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)

        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)
        disc_generated_output = Dis(fake_hic_h, training=False)

        loss_high_0 = generator_bce_loss(disc_generated_output)
        loss_high_1 = generator_mse_loss(fake_hic_h, imgr_filter)
        loss_high_2 = generator_ssim_loss(fake_hic_h, imgr_filter)

        loss_high = loss_high_0 * loss_weights[0] + \
                    loss_high_1 * loss_weights[1] + \
                    loss_high_2 * loss_weights[2]

    gen_low_v = []
    gen_low_v += Gen.get_layer('dsd_x2').trainable_variables
    gen_low_v += Gen.get_layer('r1e_x2').trainable_variables
    gen_low_v += Gen.get_layer('dsd_x4').trainable_variables
    gen_low_v += Gen.get_layer('r1e_x4').trainable_variables
    gradients_of_generator_low = x.gradient(loss_low, gen_low_v)

    gen_high_v = []
    gen_high_v += Gen.get_layer('r1c_x2').trainable_variables
    gen_high_v += Gen.get_layer('usc_x2').trainable_variables
    gen_high_v += Gen.get_layer('r1c_x4').trainable_variables
    gen_high_v += Gen.get_layer('usc_x4').trainable_variables
    gen_high_v += Gen.get_layer('cc_x2').trainable_variables
    gen_high_v += Gen.get_layer('sum_high').trainable_variables
    gen_high_v += Gen.get_layer('out_high').trainable_variables
    gradients_of_generator_high = gen_tape_high.gradient(loss_high, gen_high_v)

    # apply gradients
    opts[0].apply_gradients(zip(gradients_of_generator_low, gen_low_v))
    opts[1].apply_gradients(zip(gradients_of_generator_high, gen_high_v))

    # log losses
    return loss_low_ssim, loss_low_mse, loss_high_0, loss_high_1, loss_high_2


# @tf.function
def _train_step_discriminator(Gen, Dis, imgl, imgr, loss_filter, opts):
    with tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=False)
        # fake_hic_h = fake_hic[3]
        fake_hic_h = fake_hic[2]

        mfilter_high = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)

        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)

        disc_generated_output = Dis(fake_hic_h, training=True)
        disc_real_output = Dis(imgr_filter, training=True)
        loss = discriminator_bce_loss(disc_real_output, disc_generated_output)
    discriminator_gradients = disc_tape.gradient(loss, Dis.trainable_variables)
    opts[0].apply_gradients(
        zip(discriminator_gradients, Dis.trainable_variables))
    return loss


@tf.function
def tracegraph(x, model):
    return model(x)


def fit(gen, dis, dataset, epochs, len_high_size,
        scale, valid_dataset=None, log_dir=None, saved_model_dir=None):
    if log_dir is None:
        log_dir = './logs/model'
    logging.basicConfig(filename=os.path.join(
        log_dir, 'training.log'), level=logging.INFO)
    if saved_model_dir is None:
        saved_model_dir = './saved_model'
    generator_optimizer_low = tf.keras.optimizers.Adam()
    generator_optimizer_high = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()
    opts = [generator_optimizer_low, generator_optimizer_high]

    # for generator#, discriminator_optimizer]
    generator_log_ssim_low = tf.keras.metrics.Mean(
        'train_gen_low_ssim_loss', dtype=tf.float32)
    generator_log_mse_low = tf.keras.metrics.Mean(
        'train_gen_low_mse_loss', dtype=tf.float32)
    generator_log_mse_high = tf.keras.metrics.Mean(
        'train_gen_high_mse_loss', dtype=tf.float32)
    generator_log_bce_high = tf.keras.metrics.Mean(
        'train_gen_high_bce_loss', dtype=tf.float32)
    generator_log_ssim_high = tf.keras.metrics.Mean(
        'train_gen_high_ssim_loss', dtype=tf.float32)
    discriminator_log = tf.keras.metrics.Mean(
        'train_discriminator_loss', dtype=tf.float32)
    if valid_dataset is not None:
        valid_gen_log_h_bce = tf.keras.metrics.Mean(
            'valid_gen_high_bce_loss', dtype=tf.float32)
        valid_gen_log_h_mse = tf.keras.metrics.Mean(
            'valid_gen_high_mse_loss', dtype=tf.float32)
        valid_gen_log_h_ssim = tf.keras.metrics.Mean(
            'valid_gen_high_ssim_loss', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = os.path.join(log_dir, current_time, 'generator')
    train_summary_G_writer = tf.summary.create_file_writer(train_log_dir)
    train_log_dir = os.path.join(log_dir, current_time, 'discriminator')
    train_summary_D_writer = tf.summary.create_file_writer(train_log_dir)

    demo_log_dir = os.path.join(log_dir, current_time, 'valid')
    demo_writer = tf.summary.create_file_writer(demo_log_dir)

    train_log_dir = os.path.join(log_dir, current_time, 'model')
    writer = tf.summary.create_file_writer(train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    # Forward pass
    tracegraph(tf.zeros((1, len_high_size, len_high_size, 1)), gen)
    with writer.as_default():
        tf.summary.trace_export(name="model_gen_trace",
                                step=0, profiler_outdir=train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)

    tracegraph(tf.zeros((1, len_high_size, len_high_size, 1)), dis)
    with writer.as_default():
        tf.summary.trace_export(name="model_dis_trace",
                                step=0, profiler_outdir=train_log_dir)

    with demo_writer.as_default():
        [_, (demo_input_low, demo_input_high)] = next(
            enumerate(dataset.take(1)))
        mpy = demo_input_low.numpy()
        m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        tf.summary.image("demo data low examples",
                         images, max_outputs=16, step=0)
        mpy = demo_input_high.numpy()
        m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        tf.summary.image("demo data high examples",
                         images, max_outputs=16, step=0)

    len_x2 = int(len_high_size/2)
    len_x4 = int(len_high_size/4)
    loss_filter_low_x2 = np.ones(shape=(len_x2, len_x2)) - \
        np.diag(np.ones(shape=(len_x2,)), k=0) - \
        np.diag(np.ones(shape=(len_x2-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_x2-1,)), k=1)
    loss_filter_low_x4 = np.ones(shape=(len_x4, len_x4)) - \
        np.diag(np.ones(shape=(len_x4,)), k=0) - \
        np.diag(np.ones(shape=(len_x4-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_x4-1,)), k=1)
    loss_filter_high = np.ones(shape=(len_high_size, len_high_size)) - \
        np.diag(np.ones(shape=(len_high_size,)), k=0) - \
        np.diag(np.ones(shape=(len_high_size-1,)), k=-1) - \
        np.diag(np.ones(shape=(len_high_size-1,)), k=1)

    [_, (demo_input_low, demo_input_high)] = next(enumerate(dataset.take(1)))

    train_step_generator = tf.function(_train_step_generator)
    train_step_discriminator = tf.function(_train_step_discriminator)
    best_loss = None
    for epoch in range(epochs):
        start = time.time()
        # train

        generator_log_ssim_low.reset_states()
        generator_log_mse_low.reset_states()
        generator_log_ssim_high.reset_states()
        generator_log_mse_high.reset_states()
        generator_log_bce_high.reset_states()

        discriminator_log.reset_states()

        if(epoch <= int(40)):
            loss_weights = [0.0, 10.0, 0.0]
        else:
            loss_weights = [0.1, 10.0, 0.0]

        for i, (low_m, high_m) in enumerate(dataset):

            g_ssim_l, g_mse_l, g_bce_h, g_mse_h, g_ssim_h = \
                train_step_generator(Gen=gen, Dis=dis,
                                        imgl=tf.dtypes.cast(
                                            low_m, tf.float32),
                                        imgr=tf.dtypes.cast(
                                            high_m, tf.float32),
                                        loss_filter=[loss_filter_low_x2, loss_filter_low_x4,
                                                    loss_filter_high],
                                        loss_weights=loss_weights,
                                        opts=opts)
            generator_log_ssim_low.update_state(g_ssim_l)
            generator_log_mse_low.update_state(g_mse_l)
            generator_log_ssim_high.update_state(g_ssim_h)
            generator_log_mse_high.update_state(g_mse_h)
            generator_log_bce_high.update_state(g_bce_h)

            # Gen, Dis, imgl, imgr, loss_filter, opts, train_logs
            d_loss = train_step_discriminator(Gen=gen, Dis=dis,
                                                imgl=tf.dtypes.cast(
                                                    low_m, tf.float32),
                                                imgr=tf.dtypes.cast(
                                                    high_m, tf.float32),
                                                loss_filter=[
                                                    loss_filter_high],
                                                opts=[discriminator_optimizer])
            discriminator_log.update_state(d_loss)

        # save model weights as checkpoints
        g_loss = g_bce_h*loss_weights[0] + g_mse_h*loss_weights[1]
        if best_loss is None or g_loss < best_loss:
            gen.save_weights(os.path.join(
                saved_model_dir, current_time, 'gen_weights_'+str(len_high_size)))
            dis.save_weights(os.path.join(
                saved_model_dir, current_time, 'dis_weights_'+str(len_high_size)))
            best_loss = g_loss

        # valid dataset
        if valid_dataset is not None:
            valid_gen_log_h_bce.reset_states()
            valid_gen_log_h_mse.reset_states()
            valid_gen_log_h_ssim.reset_states()
            for i, (low_m, high_m) in enumerate(valid_dataset):
                [dpl_x2, dpl_x4, dph, _, _] = gen(low_m, training=False)
                mfilter_high = tf.expand_dims(loss_filter_high, axis=0)
                mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
                mfilter_high = tf.cast(mfilter_high, tf.float32)
                fake_hic_h = tf.multiply(dph, mfilter_high)
                imgr_filter = tf.multiply(high_m, mfilter_high)
                disc_generated_output = dis(fake_hic_h, training=False)

                valid_gen_log_h_bce.update_state(generator_bce_loss(disc_generated_output))
                valid_gen_log_h_mse.update_state(generator_mse_loss(fake_hic_h, imgr_filter))
                valid_gen_log_h_ssim.update_state(generator_ssim_loss(fake_hic_h, imgr_filter))


        [dpl_x2, dpl_x4, dph, _, _] = gen(demo_input_low, training=False)
        demo_disc_generated = dis(dph, training=False)
        demo_disc_true = dis(demo_input_high, training=False)

        with train_summary_G_writer.as_default():
            tf.summary.scalar('loss_gen_low_disssim',
                              generator_log_ssim_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_low_mse',
                              generator_log_mse_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_mse',
                              generator_log_mse_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_disssim',
                              generator_log_ssim_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_bce',
                              generator_log_bce_high.result(), step=epoch)
            if valid_dataset is not None:
                tf.summary.scalar('valid_gen_high_bce_loss',
                                  valid_gen_log_h_bce.result(), step=epoch)
                tf.summary.scalar('valid_gen_high_mse_loss',
                                  valid_gen_log_h_mse.result(), step=epoch)
                tf.summary.scalar('valid_gen_high_disssim_loss',
                                  valid_gen_log_h_ssim.result(), step=epoch)
            mpy = dpl_x2.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_low_x2', data=image, step=epoch)
            mpy = dpl_x4.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_low_x4', data=image, step=epoch)
            mpy = dph.numpy()
            m = np.log1p(1000*np.squeeze(mpy[:, :, :, 0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_high', data=image, step=epoch)

        with train_summary_D_writer.as_default():
            tf.summary.scalar(
                'loss_dis', discriminator_log.result(), step=epoch)
            mpy = demo_disc_generated.numpy()
            m = np.squeeze(mpy).reshape((3, 3))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_gen', data=image, step=epoch)
            mpy = demo_disc_true.numpy()
            m = np.squeeze(mpy).reshape((3, 3))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_true', data=image, step=epoch)
        if valid_dataset is not None:
            logging.info('Time for epoch [{}/{}] is {:.2f} sec. [Training, mse: {:.6f}, dis-ssim {:.6f} ][Valid, mse: {:.6f}, dis-ssim: {:.6f}]'.format(
                epoch + 1, epochs, time.time()-start,
                generator_log_mse_high.result(), generator_log_ssim_high.result(),
                valid_gen_log_h_mse.result(), valid_gen_log_h_ssim.result()))
        else:
            logging.info('Time for epoch [{}/{}] is {:.2f} sec. [Training, mse: {:.6f}, dis-ssim {:.6f} ]'.format(
                epoch + 1, epochs, time.time()-start,
                generator_log_mse_high.result(), generator_log_ssim_high.result()))


def plot_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 10))
    if len(m.shape) == 3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3, 3, i+1)
            ax.matshow(np.squeeze(m[i, :, :]), cmap='RdBu_r')
        plt.tight_layout()
    else:
        plt.matshow(m, cmap='RdBu_r')
        plt.colorbar()
        plt.tight_layout()
    return figure


def plot_prob_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    figure = plt.figure(figsize=(10, 10))

    m = 1 / (1 + np.exp(-m))
    if len(m.shape) == 3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3, 3, i+1)
            im = ax.matshow(np.squeeze(m[i, :, :]), cmap='RdBu_r')
            txt = "mean prob is {:5.4f}".format(np.mean(m[i, :, :]))
            ax.set_title(txt)
            im.set_clim(0.001, 1.001)
        plt.tight_layout()
    else:
        ax = figure.subplots()
        im = ax.matshow(m, cmap='RdBu_r', clim=[0.0, 1.0])
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'))
        for (i, j), z in np.ndenumerate(m):
            ax.text(j, i, '{:2.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    return figure


def plot_to_image(figure):
    import io
    import matplotlib.pyplot as plt
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__ == '__main__':
    len_size = 400
    scale = 4
    Gen = make_generator_model(len_high_size=len_size, scale=scale)
    Dis = make_discriminator_model(len_high_size=len_size, scale=scale)
    print(Gen.summary())
    tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
    print(Dis.summary())
    tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)
