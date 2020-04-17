import numpy as np
import tensorflow as tf


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


class Sum_R1M(tf.keras.layers.Layer):
    def __init__(self, name='SR1M'):
        super(Sum_R1M, self).__init__(name=name)

    def call(self, input):
        return tf.reduce_sum(input, axis=-1, keepdims=True)


class Normal(tf.keras.layers.Layer):
    def __init__(self, input_dim, name='DW'):
        super(Normal, self).__init__(name=name)
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, input_dim, 1, 1), dtype='float32'), trainable=True)
        d_init = tf.zeros_initializer()
        self.d = tf.Variable(initial_value=d_init(
            shape=(1, input_dim), dtype='float32'), trainable=True)

    def call(self, inputs):
        rowsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=1, keepdims=True))
        colsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=2, keepdims=True))
        sumele = tf.math.multiply(rowsr, colsr)
        #tf.math.divide_no_nan(inputs, sumele)
        Div = tf.math.divide_no_nan(inputs, sumele)
        self.w.assign(tf.nn.relu(self.w))
        self.d.assign(tf.nn.relu(self.d))
        WT = tf.transpose(self.w, perm=[0, 2, 1, 3])
        M = tf.multiply(self.w, WT)
        opd = tf.linalg.LinearOperatorToeplitz(self.d, self.d)
        opd = tf.expand_dims(opd.to_dense(), axis=-1)
        return tf.add(tf.multiply(Div, M), opd)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    result.add(tf.keras.layers.MaxPool2D())
    return result


def make_generator_model():
    In = tf.keras.layers.Input(
        shape=(128, 128, 1), name='In', dtype=tf.float32)
    Decl = tf.keras.layers.Conv2D(32, [1, 128], strides=1, padding='valid', data_format="channels_last", activation='relu', use_bias=False,
                                  kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1),  name='Decl')(In)

    WeiR1Ml = Weight_R1M(name='WR1Ml')(Decl)
    Recl = Reconstruct_R1M(32, name='Recl')(WeiR1Ml)
    Suml = Sum_R1M(name='Suml')(Recl)
    low_out = Normal(128, name='Out_low')(Suml)

    up_o = tf.keras.layers.UpSampling2D(
        size=(4, 4), data_format='channels_last', name='Upo')(In)
    m_F = tf.constant(1/16.0, shape=(1, 1, 1, 1))
    up_o = tf.keras.layers.Multiply()([up_o, m_F])

    up_1 = tf.keras.layers.UpSampling2D(
        size=(4, 1), data_format='channels_last', name='UpSample')(WeiR1Ml)
    m_F = tf.constant(1/4.0, shape=(1, 1, 1, 1))
    up_1 = tf.keras.layers.Multiply()([up_1, m_F])
    #trans_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(17,1), strides=(1,1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT1')(up_1)
    #trans_2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(5,1), strides=(2,1), padding='same', data_format="channels_last", activation='relu', use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), name='C2DT2')(trans_1)
    Rech = Reconstruct_R1M(32, name='Rech')(up_1)
    # DepthwiseConv2D or SeparableConv2D https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/
    trans_1_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(
        1, 1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT1_1')(Rech)
    trans_1_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(
        1, 1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT1_2')(Rech)
    #trans_2_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5,5), strides=(1,1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT2_1')(Rech)
    #trans_2_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5,5), strides=(1,1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT2_2')(Rech)
    #trans_3_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(7,7), strides=(1,1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT3_1')(Rech)
    #trans_3_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(7,7), strides=(1,1), padding='same', data_format="channels_last", activation='relu', use_bias=True, name='C2DT3_2')(Rech)
    # , trans_2_1, trans_2_2, trans_3_1, trans_3_2])
    Concat = tf.keras.layers.concatenate([trans_1_1, trans_1_2])
    WeiR1Mh = Weight_R1M(name='WR1Mh')(Concat)
    Sumh = Sum_R1M(name='Sumh')(WeiR1Mh)
    high_out = Normal(512, name='Out_high')(Sumh)

    model = tf.keras.models.Model(
        inputs=[In], outputs=[low_out, high_out, up_o])
    return model


def make_discriminator_model():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[512, 512, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[512, 512, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    #down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)
    conv = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    '''batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)'''
    return tf.keras.Model(inputs=[inp, tar], outputs=conv)


Gen = make_generator_model()
Dis = make_discriminator_model()
print(Gen.summary())
tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
print(Dis.summary())
tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_KL_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_ssim_loss(y_pred, y_true):
    # return cross_entropy(tf.ones_like(fake_output), fake_output)
    return 1 - tf.image.ssim(y_pred, y_true, max_val=1.0)


def generator_KL_loss(d_pred):
    gan_loss = loss_object(tf.ones_like(d_pred), d_pred)
    return gan_loss


generator_optimizer_low = tf.keras.optimizers.Adam()
generator_optimizer_high = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adagrad()
@tf.function
def train_step(imgl, imgr):
    with tf.GradientTape() as gen_tape_low, tf.GradientTape() as gen_tape_high, tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=True)
        fake_hic_l = fake_hic[0]
        fake_hic_h = fake_hic[1]
        img_l_h = fake_hic[2]
        gen_loss_low = generator_lowr_loss(fake_hic_l, imgl)
        #gen_low_v = Gen.trainable_variables
        gen_low_v = []
        gen_low_v += Gen.get_layer('Decl').trainable_variables
        gen_low_v += Gen.get_layer('WR1Ml').trainable_variables
        gen_low_v += Gen.get_layer('Recl').trainable_variables
        gen_low_v += Gen.get_layer('Out_low').trainable_variables
        gradients_of_generator_low = gen_tape_low.gradient(
            gen_loss_low, gen_low_v)
        generator_optimizer_low.apply_gradients(
            zip(gradients_of_generator_low, gen_low_v))

        #disc_generated_output = Dis([fake_hic_h, imgr], training=True)

        # + generator_highr_loss(disc_generated_output)
        gen_loss_high = generator_lowr_loss(fake_hic_h, imgr)
        gen_high_v = []
        gen_high_v += Gen.get_layer('Rech').trainable_variables
        gen_high_v += Gen.get_layer('C2DT1_1').trainable_variables
        gen_high_v += Gen.get_layer('C2DT1_2').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT2_1').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT2_2').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT3_1').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT3_2').trainable_variables
        gen_high_v += Gen.get_layer('WR1Mh').trainable_variables
        gen_high_v += Gen.get_layer('Out_high').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT2').trainable_variables
        #gen_high_v += Gen.get_layer('Rech').trainable_variables
        gradients_of_generator_high = gen_tape_high.gradient(
            gen_loss_high, gen_high_v)
        generator_optimizer_high.apply_gradients(
            zip(gradients_of_generator_high, gen_high_v))

        '''disc_real_output = Dis([img_l_h, imgr], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        discriminator_gradients = disc_tape.gradient(disc_loss, Dis.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, Dis.trainable_variables))'''


def train(dataset, epochs, BATCH_SIZE):
    # print(dataset)
    for epoch in range(epochs):
        start = time.time()
        for i, (low_m, high_m) in enumerate(dataset.take(1)):
            train_step(tf.dtypes.cast(low_m, tf.float32),
                       tf.dtypes.cast(high_m, tf.float32))

        # Save the model every 15 epochs
        if (epoch + 1) % 25 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(Gen, epoch + 1, None)
            #checkpoint.save(file_prefix = checkpoint_prefix)
        print('Time for epoch {} is {} sec.'.format(
            epoch + 1, time.time()-start))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    #generate_and_save_images(Gen, epochs, None)
