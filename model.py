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
        '''d_init = tf.zeros_initializer()
        self.d = tf.Variable(initial_value=d_init(
            shape=(1, input_dim), dtype='float32'), trainable=True)'''

    def call(self, inputs):
        rowsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=1, keepdims=True))
        colsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=2, keepdims=True))
        sumele = tf.math.multiply(rowsr, colsr)
        #tf.math.divide_no_nan(inputs, sumele)
        Div = tf.math.divide_no_nan(inputs, sumele)
        self.w.assign(tf.nn.relu(self.w))
        #self.d.assign(tf.nn.relu(self.d))
        WT = tf.transpose(self.w, perm=[0, 2, 1, 3])
        M = tf.multiply(self.w, WT)
        #opd = tf.linalg.LinearOperatorToeplitz(self.d, self.d)
        #opd = tf.expand_dims(opd.to_dense(), axis=-1)
        #return tf.add(tf.multiply(Div, M), opd)
        return tf.multiply(Div, M)

class symmetry_constraints(tf.keras.constraints.Constraint):
    def __call__(self, w): 
        #for conv2d the shape of kernel = [W, H, C, K] C:channels, K:output number of filters
        Tw = tf.transpose(w, perm=[1,0,2,3])
        return (w + Tw)/2.0

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                                kernel_initializer=initializer, 
                                kernel_constraint=symmetry_constraints(), 
                                use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


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

    '''up_1 = tf.keras.layers.UpSampling2D(size=(4, 1), data_format='channels_last', name='upsample_low_1')(WeiR1Ml)
    m_F = tf.constant(1/4.0, shape=(1, 1, 1, 1))
    up_1 = tf.keras.layers.Multiply(name='scale_value_high')([up_1, m_F])
    Rech = Reconstruct_R1M(1024, name='rec_high')(up_1)
    paddings = tf.constant([[0,0],[2, 2], [2, 2], [0,0]])
    Rech = tf.pad(Rech, paddings, "SYMMETRIC")
    trans_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5),
                                    strides=(1,1), padding='valid',
                                    data_format="channels_last",
                                    kernel_constraint=symmetry_constraints(), 
                                    activation='relu', use_bias=False, name='C2DT1')(Rech)
    batchnorm_1 = tf.keras.layers.BatchNormalization()(trans_1)
    paddings = tf.constant([[0,0],[1, 1], [1, 1], [0,0]])
    batchnorm_1 = tf.pad(batchnorm_1, paddings, "SYMMETRIC")
    trans_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                                    strides=(1,1), padding='valid',
                                    data_format="channels_last",
                                    kernel_constraint=symmetry_constraints(),
                                    activation='relu', use_bias=False, name='C2DT2')(batchnorm_1)
    Sumh = Sum_R1M(name='sum_high')(trans_2)
    high_out = Normal(int(len_low_size*scale), name='out_high')(Sumh)'''

    
    Rech = Reconstruct_R1M(1024, name='rec_high')(WeiR1Ml)
    paddings = tf.constant([[0,0],[1, 1], [1, 1], [0,0]])
    Rech = tf.pad(Rech, paddings, "SYMMETRIC")
    trans_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                                    strides=(1,1), padding='valid',
                                    data_format="channels_last",
                                    kernel_constraint=symmetry_constraints(),
                                    activation='relu', use_bias=False, name='C2DT1')(Rech)
    batchnorm_1 = tf.keras.layers.BatchNormalization()(trans_1)
    up_1 = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear' , name='upsample_low_1')(batchnorm_1)
    m_F = tf.constant(1/16.0, shape=(1, 1, 1, 1))
    up_1 = tf.keras.layers.Multiply(name='scale_value_high')([up_1, m_F])
    paddings = tf.constant([[0,0],[1, 1], [1, 1], [0,0]])
    batchnorm_1 = tf.pad(up_1, paddings, "SYMMETRIC")
    trans_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3),
                                    strides=(1,1), padding='valid',
                                    data_format="channels_last",
                                    kernel_constraint=symmetry_constraints(),
                                    activation='relu', use_bias=False, name='C2DT2')(batchnorm_1)
    Sumh = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1),
                                    strides=(1,1), padding='same',
                                    data_format="channels_last",
                                    kernel_constraint=tf.keras.constraints.NonNeg(),
                                    activation='relu', use_bias=False, name='sum_high')(trans_2)
    high_out = Normal(int(len_low_size*scale), name='out_high')(Sumh)

    model = tf.keras.models.Model(
        inputs=[In], outputs=[low_out, high_out, up_o])
    return model


def make_discriminator_model(len_low_size=16, scale=4):
    len_high_size = int(len_low_size*scale)
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[len_high_size, len_high_size, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    #x = inp
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  kernel_constraint=symmetry_constraints(), 
                                  use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_constraint=symmetry_constraints(), 
                                    kernel_initializer=initializer)(zero_pad2)
    #return tf.keras.Model(inputs=[inp, tar], outputs=last)
    return tf.keras.Model(inputs=inp, outputs=last)

def discriminator_KL_loss(real_output, fake_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
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

def generator_KL_loss(d_pred):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(d_pred), d_pred)
    return gan_loss


@tf.function
def train_step_generator(Gen, Dis, imgl, imgr, loss_filter, opts, train_logs):
    with tf.GradientTape() as gen_tape_low, tf.GradientTape() as gen_tape_high, tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=True)
        fake_hic_l = fake_hic[0]
        fake_hic_h = fake_hic[1]
        img_l_h = fake_hic[2]

        mfilter_low = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l = tf.multiply(fake_hic_l, mfilter_low)
        imgl_filter = tf.multiply(imgl, mfilter_low)

        mfilter_high = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)
        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        img_l_h = tf.multiply(img_l_h, mfilter_high)
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
        disc_generated_output = Dis([img_l_h, fake_hic_h], training=False)
        #disc_generated_output = Dis(fake_hic_h, training=False)
        gen_high_v = []
        gen_high_v += Gen.get_layer('rec_high').trainable_variables
        #gen_high_v += Gen.get_layer('C2DT0').trainable_variables
        #gen_high_v += Gen.get_layer('batch_normalization').trainable_variables
        gen_high_v += Gen.get_layer('C2DT1').trainable_variables
        gen_high_v += Gen.get_layer('batch_normalization').trainable_variables
        gen_high_v += Gen.get_layer('C2DT2').trainable_variables
        gen_high_v += Gen.get_layer('sum_high').trainable_variables
        gen_high_v += Gen.get_layer('out_high').trainable_variables
        gen_loss_high_0 = generator_mse_loss(fake_hic_h, imgr_filter)
        gen_loss_high_1 = generator_KL_loss(disc_generated_output)
        gen_loss_high_2 = generator_ssim_loss(fake_hic_h, imgr_filter)
        gen_loss_high = gen_loss_high_0*10+ gen_loss_high_2*10 + gen_loss_high_1
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
        img_l_h = fake_hic[2]
        mfilter_high = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)
        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        img_l_h = tf.multiply(img_l_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)
        disc_generated_output = Dis([img_l_h, fake_hic_h], training=True)
        disc_real_output = Dis([img_l_h, imgr_filter], training=True)
        #disc_generated_output = Dis(fake_hic_h, training=True)
        #disc_real_output = Dis(imgr_filter, training=True)
        disc_loss = discriminator_KL_loss( disc_real_output, disc_generated_output)
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
    generator_log_kl_high = tf.keras.metrics.Mean('train_gen_high_KL_loss', dtype=tf.float32)
    generator_log_ssim_high = tf.keras.metrics.Mean('train_gen_high_ssim_loss', dtype=tf.float32)
    discriminator_log = tf.keras.metrics.Mean('train_discriminator_loss', dtype=tf.float32)
    logs = [generator_log_ssim_low, generator_log_mse_low, generator_log_mse_high, generator_log_kl_high, generator_log_ssim_high]# for generator, discriminator_log]
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
            train_step_generator(gen, dis, 
                                tf.dtypes.cast(low_m, tf.float32), tf.dtypes.cast(high_m, tf.float32),
                                [loss_filter_low, loss_filter_high],
                                opts, logs)
            if((epoch+1) % 2):
                train_step_discriminator(gen, dis, 
                                tf.dtypes.cast(low_m, tf.float32), tf.dtypes.cast(high_m, tf.float32),
                                [loss_filter_low, loss_filter_high],
                                [discriminator_optimizer], [discriminator_log])
        # log the model epochs
        [demo_pred_low, demo_pred_high, demo_up] = gen(demo_input_low, training=False)
        demo_disc_generated = dis([demo_pred_high, demo_up], training=False)
        demo_disc_true = dis([demo_input_high, demo_up], training=False)
        with train_summary_G_writer.as_default():
            tf.summary.scalar('loss_gen_low_disssim', generator_log_ssim_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_low_mse', generator_log_mse_low.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_mse', generator_log_mse_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_kl', generator_log_kl_high.result(), step=epoch)
            tf.summary.scalar('loss_gen_high_disssim', generator_log_ssim_high.result(), step=epoch)
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
            m = np.squeeze(mpy[:,:,:,0])
            fig = plot_matrix(m)
            image = plot_to_image(fig)
            tf.summary.image(name='dis_gen', data=image, step=epoch)
            mpy = demo_disc_true.numpy()
            m = np.squeeze(mpy[:,:,:,0])
            fig = plot_matrix(m)
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
