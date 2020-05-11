import numpy as np
import tensorflow as tf
import time
from IPython import display
import datetime
class Reconstruct_R1M(tf.keras.layers.Layer):
    def __init__(self, filters, name='RR'):
        super(Reconstruct_R1M, self).__init__(name=name)
        self.num_outputs = filters
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, 1, 1, filters), dtype='float32'))

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


class Subpixel(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, [bsize, a, b, tf.cast(c/(r*r), tf.int32), r, r]) # bsize, a, b, c/(r*r), r, r
        #X = tf.keras.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        X = tf.transpose(X, perm=[0,1,2,5,4,3])
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = tf.keras.backend.concatenate(X, axis=2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = tf.keras.backend.concatenate(X, axis=2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], unshifted[3]/(self.r*self.r))

    def get_config(self):
        config = super(tf.keras.layers.Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=self.r*self.r
        config['r'] = self.r
        return config


class Sum_R1M(tf.keras.layers.Layer):
    def __init__(self, name='SR1M'):
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
        low = tf.transpose(up, perm=[0,2,1,3])
        return up + low

class Normal(tf.keras.layers.Layer):
    def __init__(self, input_dim, name='DW'):
        super(Normal, self).__init__(name=name)
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, input_dim, 1, 1), dtype='float32'), trainable=True)

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

'''class symmetry_constraints(tf.keras.constraints.Constraint):
    def __call__(self, w): 
        #for conv2d the shape of kernel = [W, H, C, K] C:channels, K:output number of filters
        Tw = tf.transpose(w, perm=[1,0,2,3])
        return (w + Tw)/2.0'''

def make_generator_model(len_low_size=16, scale=4):
    In = tf.keras.layers.Input(
        shape=(len_low_size, len_low_size, 1), name='in', dtype=tf.float32)
    Decl = tf.keras.layers.Conv2D(1024, [1, len_low_size], strides=1, padding='valid', data_format="channels_last", 
                                    activation='relu', use_bias=False,
                                    kernel_constraint=tf.keras.constraints.NonNeg(), 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1), 
                                    name='dec_low')(In)

    WeiR1Ml = Weight_R1M(name='WR1Ml')(Decl)
    Recl = Reconstruct_R1M(1024, name='rec_low')(WeiR1Ml)
    Suml = Sum_R1M(name='sum_low')(Recl)
    low_out = Normal(len_low_size, name='out_low')(Suml)

    up_o = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', name='up_in')(In)
    m_F = tf.constant(1/16.0, shape=(1, 1, 1, 1))
    up_o = tf.keras.layers.Multiply(name='scale_value_in')([up_o, m_F])

    Rech = Reconstruct_R1M(1024, name='rec_high')(WeiR1Ml)
    trans_1 = Subpixel(filters= int(128), kernel_size=(3,3), r=2, 
                        activation='relu', use_bias=False, padding='same', 
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1), 
                        name='subpixel_1')(Rech)
    sym = Symmetry_R1M(name='SYM_1')(trans_1)
    trans_2 = Subpixel(filters= int(128), kernel_size=(3,3), r=2, 
                        activation='relu', use_bias=False, padding='same', 
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1), 
                        name='subpixel_2')(sym)
    sym = Symmetry_R1M(name='SYM_2')(trans_2)
    Sumh = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1),
                                    strides=(1,1), padding='same',
                                    data_format="channels_last",
                                    kernel_constraint=tf.keras.constraints.NonNeg(),
                                    activation='relu', use_bias=False, name='sum_high')(sym)
    high_out = Normal(int(len_low_size*scale), name='out_high')(Sumh)

    model = tf.keras.models.Model(inputs=[In], outputs=[low_out, high_out, up_o])
    return model

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid', 
                                kernel_initializer=initializer, 
                                use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

"""def make_discriminator_model(len_low_size=16, scale=4):
    '''PatchGAN 1 pixel of output represents X pixels of input: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
     The "70" is implicit, it's not written anywhere in the code but instead emerges as a mathematical consequence of the network architecture.
    The math is here: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    compute input size from a given output size:
    f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize; fix output_size as 1 
    '''
    len_high_size = int(len_low_size*scale)
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='input_image')
    #tar = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='target_image')
    #x = tf.keras.layers.concatenate([inp, tar])
    x = inp
    down1 = downsample(256, 3, False)(x)
    down2 = downsample(512, 3)(down1)
    #down3 = downsample(256, 3)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)
    conv = tf.keras.layers.Conv2D(512, 3, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=True)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    #zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 3, strides=1, padding='valid', kernel_initializer=initializer,use_bias=True)(leaky_relu)
    last = tf.keras.layers.Activation('sigmoid')(last)
    #return tf.keras.Model(inputs=[inp, tar], outputs=last)
    return tf.keras.Model(inputs=inp, outputs=last)"""

"""def make_discriminator_model(len_low_size=16, scale=4):
    '''PatchGAN 1 pixel of output represents X pixels of input: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
     The "70" is implicit, it's not written anywhere in the code but instead emerges as a mathematical consequence of the network architecture.
    The math is here: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    compute input size from a given output size:
    f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize; fix output_size as 1 
    '''
    len_high_size = int(len_low_size*scale)
    initializer = tf.random_normal_initializer(0., 0.2)
    inp = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='input_image')
    dec = tf.keras.layers.Conv2D(1024, [1, len_high_size], strides=1, padding='valid', data_format="channels_last", 
                                    use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(0., 0.01), 
                                    name='dec')(inp)
    batchnorm = tf.keras.layers.BatchNormalization()(dec)

    conv = tf.keras.layers.Conv2D(128, [3, 1], strides=[2,1], padding='valid', data_format="channels_last", 
                                    activation='relu', use_bias=True,
                                    kernel_initializer=initializer, 
                                    )(batchnorm)
    batchnorm = tf.keras.layers.BatchNormalization()(conv)
    #leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm)

    conv = tf.keras.layers.Conv2D(128, [3, 1], strides=[2,1], padding='valid', data_format="channels_last", 
                                    activation='relu', use_bias=True,
                                    kernel_initializer=initializer, 
                                    )(batchnorm)
    batchnorm = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(1, [1, 1], strides=[1,1], padding='same', data_format="channels_last", 
                                    activation=None, use_bias=True,
                                    kernel_initializer=initializer, 
                                    )(batchnorm)

    last = tf.keras.layers.Flatten()(conv)
    last = tf.keras.layers.Dense(1, activation='sigmoid')(last)
    #last = tf.keras.layers.Reshape((31, 32))(last)
    return tf.keras.Model(inputs=inp, outputs=last)"""

"""def make_discriminator_model(len_low_size=16, scale=4):
    len_high_size = int(len_low_size*scale)
    inp = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='input_image')

    zero_pad = tf.keras.layers.ZeroPadding2D()(inp)
    conv = tf.keras.layers.Conv2D(64, 4, strides=2, padding='valid', use_bias=False)(zero_pad)
    sym = Symmetry_R1M()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(sym)

    zero_pad = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    conv = tf.keras.layers.Conv2D(128, 4, strides=2, padding='valid', use_bias=False)(zero_pad)
    sym = Symmetry_R1M()(conv)
    batchnorm = tf.keras.layers.BatchNormalization()(sym)
    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm)

    '''zero_pad = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    conv = tf.keras.layers.Conv2D(256, 4, strides=2, padding='valid', use_bias=False)(zero_pad)
    batchnorm = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm)'''

    zero_pad = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    conv = tf.keras.layers.Conv2D(512, 4, strides=2, padding='valid', use_bias=False)(zero_pad)
    sym = Symmetry_R1M()(conv)
    batchnorm = tf.keras.layers.BatchNormalization()(sym)
    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm)

    last = tf.keras.layers.Conv2D(1, 1, strides=1, padding='valid', use_bias=False, activation='sigmoid')(leaky_relu)
    #last = tf.keras.layers.Flatten()(last)
    #last = tf.keras.layers.Dense(1, activation='sigmoid')(last)
    return tf.keras.Model(inputs=inp, outputs=last)"""

def make_discriminator_model(len_low_size=16, scale=4):
    len_high_size = int(len_low_size*scale)
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    nin = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1])
    n = tf.keras.layers.Conv2D(df_dim, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(nin)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 2, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 4, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 8, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 16, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    '''n = tf.keras.layers.Conv2D(df_dim * 32, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 16, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)'''
    n = tf.keras.layers.Conv2D(df_dim * 8, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    nn = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    nn = tf.keras.layers.LeakyReLU(0.2)(nn)

    n = tf.keras.layers.Conv2D(df_dim * 2, (1, 1), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init)(nn)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 2, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = Symmetry_R1M()(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Conv2D(df_dim * 8, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.LeakyReLU(0.2)(n)
    n = tf.keras.layers.Add()([n, nn])
    n = tf.keras.layers.LeakyReLU(0.2)(n)

    n = tf.keras.layers.Flatten()(n)
    no = tf.keras.layers.Dense(1, kernel_initializer=w_init, activation='sigmoid')(n)
    D = tf.keras.Model(inputs=nin, outputs=no)
    #D = Model(inputs=nin, outputs=no)
    return D

def discriminator_bce_loss(real_output, fake_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_ssim_loss(y_pred, y_true):#, m_filter):
    return (1 - tf.image.ssim(y_pred, y_true, max_val=1.0))/2.0

def generator_mse_loss(y_pred, y_true):#, m_filter):
    diff = tf.math.squared_difference(y_pred, y_true)
    s = tf.reduce_sum(diff, axis=-1)
    s = tf.reduce_sum(s, axis=-1)
    s = tf.reduce_mean(s, axis=-1)
    return s

def generator_bce_loss(d_pred):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    gan_loss = loss_object(tf.ones_like(d_pred), d_pred)
    return gan_loss


@tf.function
def train_step_generator(Gen, Dis, imgl, imgr, loss_filter, loss_weights, opts, train_logs):
    with tf.GradientTape() as gen_tape_low, tf.GradientTape() as gen_tape_high, tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=True)
        fake_hic_l = fake_hic[0]
        fake_hic_h = fake_hic[1]
        #img_l_h = fake_hic[2]

        mfilter_low = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l = tf.multiply(fake_hic_l, mfilter_low)
        imgl_filter = tf.multiply(imgl, mfilter_low)

        mfilter_high = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)
        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        #img_l_h = tf.multiply(img_l_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)
        #gen_low_v = Gen.trainable_variables
        gen_low_v = []
        gen_low_v += Gen.get_layer('dec_low').trainable_variables
        gen_low_v += Gen.get_layer('WR1Ml').trainable_variables
        gen_low_v += Gen.get_layer('rec_low').trainable_variables
        gen_low_v += Gen.get_layer('out_low').trainable_variables

        gen_loss_low_ssim = generator_ssim_loss(fake_hic_l, imgl_filter)
        gen_loss_low_mse = generator_mse_loss(fake_hic_l, imgl_filter)
        gen_loss_low = gen_loss_low_ssim + gen_loss_low_mse
        gradients_of_generator_low = gen_tape_low.gradient(gen_loss_low, gen_low_v)
        opts[0].apply_gradients(zip(gradients_of_generator_low, gen_low_v))
        train_logs[0](gen_loss_low_ssim)
        train_logs[1](gen_loss_low_mse)
        #if(epoch_flag):
        #disc_generated_output = Dis([img_l_h, fake_hic_h], training=False)
        disc_generated_output = Dis(fake_hic_h, training=False)
        gen_high_v = []
        gen_high_v += Gen.get_layer('rec_high').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT0').trainable_variables
        #gen_high_v += Gen.get_layer('batch_normalization').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT1').trainable_variables
        #gen_high_v += Gen.get_layer('batch_normalization').trainable_variables
        gen_high_v += Gen.get_layer('subpixel_1').trainable_variables
        gen_high_v += Gen.get_layer('subpixel_2').trainable_variables
        #gen_high_v += Gen.get_layer('batch_normalization_1').trainable_variables
        gen_high_v += Gen.get_layer('sum_high').trainable_variables
        gen_high_v += Gen.get_layer('out_high').trainable_variables
        gen_loss_high_0 = generator_bce_loss(disc_generated_output) 
        gen_loss_high_1 = generator_mse_loss(fake_hic_h, imgr_filter)
        gen_loss_high_2 = generator_ssim_loss(fake_hic_h, imgr_filter)
        gen_loss_high = gen_loss_high_0*loss_weights[0]+ gen_loss_high_1*loss_weights[1] + gen_loss_high_2*loss_weights[2]
        gradients_of_generator_high = gen_tape_high.gradient(gen_loss_high, gen_high_v)
        opts[1].apply_gradients(zip(gradients_of_generator_high, gen_high_v))
        train_logs[2](gen_loss_high_0)
        train_logs[3](gen_loss_high_1)
        train_logs[4](gen_loss_high_2)


@tf.function
def train_step_discriminator(Gen, Dis, imgl, imgr, loss_filter, opts, train_logs):
    with tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=False)
        fake_hic_l = fake_hic[0]
        fake_hic_h = fake_hic[1]
        #img_l_h = fake_hic[2]
        mfilter_high = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)
        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        #img_l_h = tf.multiply(img_l_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)
        #disc_generated_output = Dis([img_l_h, fake_hic_h], training=True)
        #disc_real_output = Dis([img_l_h, imgr_filter], training=True)
        disc_generated_output = Dis(fake_hic_h, training=True)
        disc_real_output = Dis(imgr_filter, training=True)
        disc_loss = discriminator_bce_loss( disc_real_output, disc_generated_output)
        discriminator_gradients = disc_tape.gradient(disc_loss, Dis.trainable_variables)
        opts[0].apply_gradients(zip(discriminator_gradients, Dis.trainable_variables))
        train_logs[0](disc_loss)

@tf.function
def tracegraph(x, model):
    return model(x)

def train(gen, dis, dataset, epochs, len_low_size, scale, test_dataset=None):
    len_high_size = int(len_low_size*scale)
    generator_optimizer_low = tf.keras.optimizers.Adam()
    generator_optimizer_high = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adagrad()
    opts = [generator_optimizer_low, generator_optimizer_high]# for generator#, discriminator_optimizer]
    generator_log_ssim_low = tf.keras.metrics.Mean('train_gen_low_ssim_loss', dtype=tf.float32)
    generator_log_mse_low = tf.keras.metrics.Mean('train_gen_low_mse_loss', dtype=tf.float32)
    generator_log_mse_high = tf.keras.metrics.Mean('train_gen_high_mse_loss', dtype=tf.float32)
    generator_log_bce_high = tf.keras.metrics.Mean('train_gen_high_bce_loss', dtype=tf.float32)
    generator_log_ssim_high = tf.keras.metrics.Mean('train_gen_high_ssim_loss', dtype=tf.float32)
    discriminator_log = tf.keras.metrics.Mean('train_discriminator_loss', dtype=tf.float32)
    logs = [generator_log_ssim_low, generator_log_mse_low, generator_log_bce_high, generator_log_mse_high, generator_log_ssim_high]# for generator, discriminator_log]
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/generator'
    train_summary_G_writer = tf.summary.create_file_writer(train_log_dir)
    train_log_dir = 'logs/gradient_tape/' + current_time + '/discriminator'
    train_summary_D_writer = tf.summary.create_file_writer(train_log_dir)
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    test_writer = tf.summary.create_file_writer(test_log_dir)

    train_log_dir = 'logs/gradient_tape/' + current_time + '/model'
    writer = tf.summary.create_file_writer(train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    # Forward pass
    tracegraph(tf.zeros((1, len_low_size, len_low_size, 1)), gen)
    with writer.as_default():
        tf.summary.trace_export(name="model_gen_trace", step=0, profiler_outdir=train_log_dir)
    tf.summary.trace_on(graph=True, profiler=False)
    tracegraph([tf.zeros((1, len_high_size, len_high_size, 1)), tf.zeros((1,len_high_size,len_high_size,1))], dis)
    with writer.as_default():
        tf.summary.trace_export(name="model_dis_trace", step=0, profiler_outdir=train_log_dir)

    with test_writer.as_default():
        [_, (test_input_low, test_input_high)] = next(enumerate(test_dataset.take(1)))
        mpy = test_input_low.numpy()
        m = np.log1p(100*np.squeeze(mpy[:,:,:,0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        #images = np.reshape(test_input_low[0:16], (-1, 32, 32, 1))
        tf.summary.image("16 training data low examples", images, max_outputs=16, step=0)
        mpy = test_input_high.numpy()
        m = np.log1p(100*np.squeeze(mpy[:,:,:,0]))
        fig = plot_matrix(m)
        images = plot_to_image(fig)
        #images = np.reshape(test_input_high[0:16], (-1, 128, 128, 1))
        tf.summary.image("16 training data high examples", images, max_outputs=16, step=0)

    loss_filter_low = np.ones(shape=(len_low_size,len_low_size)) - np.diag(np.ones(shape=(len_low_size,)), k=0) - np.diag(np.ones(shape=(len_low_size-1,)), k=-1) - np.diag(np.ones(shape=(len_low_size-1,)), k=1)
    loss_filter_high = np.ones(shape=(len_high_size,len_high_size)) - np.diag(np.ones(shape=(len_high_size,)), k=0) - np.diag(np.ones(shape=(len_high_size-1,)), k=-1) - np.diag(np.ones(shape=(len_high_size-1,)), k=1)

    [_, (demo_input_low, demo_input_high)] = next(enumerate(test_dataset.take(1)))
    for epoch in range(epochs):
        start = time.time()
        for i, (low_m, high_m) in enumerate(dataset):
            if(epoch<400):
                loss_weights = [0.0, 1.0, 1.0]
            else:
                loss_weights = [1.0, 10.0, 10.0]
            if(epoch<400):
                train_step_generator(gen, dis, 
                                    tf.dtypes.cast(low_m, tf.float32), tf.dtypes.cast(high_m, tf.float32),
                                    [loss_filter_low, loss_filter_high], loss_weights,
                                    opts, logs)
            if(epoch>0):
                train_step_discriminator(gen, dis, 
                                tf.dtypes.cast(low_m, tf.float32), tf.dtypes.cast(high_m, tf.float32),
                                [loss_filter_low, loss_filter_high],
                                [discriminator_optimizer], [discriminator_log])
        # log the model epochs
        [demo_pred_low, demo_pred_high, demo_up] = gen(demo_input_low, training=False)
        #demo_disc_generated = dis([demo_pred_high, demo_up], training=False)
        #demo_disc_true = dis([demo_input_high, demo_up], training=False)
        demo_disc_generated = dis(demo_pred_high, training=False)
        demo_disc_true = dis(demo_input_high, training=False)
        with train_summary_G_writer.as_default():
            tf.summary.scalar('loss_gen_low_disssim', generator_log_ssim_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_low_mse', generator_log_mse_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_mse', generator_log_mse_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_disssim', generator_log_ssim_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_bce', generator_log_bce_high.result(), step=epoch)
            mpy = demo_pred_low.numpy()
            m = np.log1p(100*np.squeeze(mpy[:,:,:,0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_low', data=image ,step=epoch)
            mpy = demo_pred_high.numpy()
            m = np.log1p(100*np.squeeze(mpy[:,:,:,0]))
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='gen_high', data=image, step=epoch)
        with train_summary_D_writer.as_default():
            tf.summary.scalar('loss_dis', discriminator_log.result(), step=epoch)
            mpy = demo_disc_generated.numpy()
            #m = np.squeeze(mpy[:,:,:,0])
            m = np.squeeze(mpy).reshape((4,4))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_gen', data=image, step=epoch)
            mpy = demo_disc_true.numpy()
            #m = np.squeeze(mpy[:,:,:,0])
            m = np.squeeze(mpy).reshape((4,4))
            fig = plot_prob_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_true', data=image, step=epoch)
        print('Time for epoch {} is {} sec.'.format(
            epoch + 1, time.time()-start))

def plot_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10,10))
    if len(m.shape)==3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3,3,i+1)
            ax.matshow(np.squeeze(m[i,:,:]), cmap='RdBu_r')
        plt.tight_layout()
    else:
        plt.matshow(m, cmap='RdBu_r')
        plt.colorbar()
        plt.tight_layout()
    return figure

def plot_prob_matrix(m):
    import numpy as np
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10,10))
    if len(m.shape)==3:
        for i in range(min(9, m.shape[0])):
            ax = figure.add_subplot(3,3,i+1)
            im = ax.matshow(np.squeeze(m[i,:,:]), cmap='RdBu_r')
            txt = "mean prob is {:5.4f}".format(np.mean(m[i,:,:]))
            ax.set_title(txt)
            im.set_clim(0.001, 1.001)
        plt.tight_layout()
    else:
        plt.matshow(m, cmap='RdBu_r')
        plt.colorbar()
        plt.clim(0.001, 1.001)
        #txt = "mean prob is {:5.4f}".format(np.mean(m[i,:,:]))
        #plt.title(txt)
        plt.tight_layout()
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
    Gen = make_generator_model(len_low_size=32, scale=4)
    Dis = make_discriminator_model(len_low_size=32, scale=4)
    print(Gen.summary())
    tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
    print(Dis.summary())
    tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

    #train(Gen, Dis, None, 0, 3)
