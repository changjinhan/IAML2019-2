import os
import tensorflow as tf
from dataloader import DataLoader, save_pianoroll_to_midi
from time import localtime, time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
ckpt_dir = 'ckpt'
best_saved_model = 'run-160-1212-1455'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# audio_dir: midi file directory, data_dir: tfrecord file directory
audio_dir = 'music_dataset'
data_dir = 'data'
sample_dir = 'sample'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
epochs = 500
batch_size = 64
noise_dim = 100
num_samples = 5
# same with Wasserstein GAN with gradient penalty parameters
n_critic = 5
lambda_gp = 10
learning_rate = 1e-4
beta_1 = 0
beta_2 = 0.99

# We will reuse this seed over training time
random_seeds = tf.random.normal([num_samples, noise_dim])

# TODO : Build your GAN model
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = tf.keras.layers.Dense(16*8, input_shape=(100,)) # DCGAN-like
        self.batch_norm_0 = tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, scale=True, center=True, trainable=True)
        self.reshape = tf.keras.layers.Reshape([16, 8, 1])

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, scale=True, center=True, trainable=True)

        self.deconv_2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, scale=True, center=True, trainable=True)

        self.deconv_3 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, scale=True, center=True, trainable=True)

        self.deconv_4 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, scale=True, center=True, trainable=True)

        self.deconv_5 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, noise, training=False):
        print('noise shape: ', noise.shape)
        x = self.dense(noise)
        print('x shape: ', x.shape)
        #x = self.batch_norm_0(x, training=training)
        #x = self.relu(x)
        x = self.reshape(x)
        print('reshape shape: ', x.shape)

        x = self.deconv_1(x)
        print('deconv1 shape: ', x.shape)
        x = self.batch_norm_1(x, training=training)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv_2(x)
        print('deconv2 shape: ', x.shape)
        x = self.batch_norm_2(x, training=training)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv_3(x)
        print('deconv3 shape: ', x.shape)
        x = self.batch_norm_3(x, training=training)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.deconv_4(x)
        print('deconv4 shape: ', x.shape)
        x = self.batch_norm_4(x, training=training)
        x = self.relu(x)
        x = self.dropout(x)

        generated_sample = self.deconv_5(x)
        print('generated shape: ', generated_sample.shape)
        generated_sample = tf.reshape(generated_sample, (generated_sample.shape[0], generated_sample.shape[1], generated_sample.shape[2]))
        print('generated reshape shape: ', generated_sample.shape)
        return generated_sample

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=1, scale=True, center=True, trainable=True)

        self.conv_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=1, scale=True, center=True, trainable=True)

        self.conv_3 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=1, scale=True, center=True, trainable=True)

        self.conv_4 = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(axis=1, scale=True, center=True, trainable=True)

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense_2 = tf.keras.layers.Dense(1)


    def call(self, x, training=False):
        print('x shape: ', x.shape)
        x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        print('x reshape shape: ', x.shape)

        x = self.conv_1(x)
        print('conv1 shape: ', x.shape)
        x = self.batch_norm_1(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv_2(x)
        print('conv2 shape: ', x.shape)
        x = self.batch_norm_2(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        print('conv3 shape: ', x.shape)
        x = self.batch_norm_3(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv_4(x)
        print('conv4 shape: ', x.shape)
        x = self.batch_norm_4(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.dense_1(x)
        print('timedistributed shape: ', x.shape)
        x = self.flatten(x)
        print('flatten shape: ', x.shape)
        decision = self.dense_2(x)
        print('decision shape: ', decision.shape)
        #decision = tf.reshape(decision, (decision.shape[0], decision.shape[1], 1, 1))
        #decision = tf.expand_dims(decision, axis=2)
        #print('decision reshape shape: ', decision.shape)
        return decision

generator = Generator()
discriminator = Discriminator()

# TODO : Construct loss function, optimizer and metrics
def wasserstein_gan_loss(real_output, fake_output):
    gen_loss = -tf.reduce_mean(fake_output)
    disc_loss = -gen_loss - tf.reduce_mean(real_output)
    return gen_loss, disc_loss

def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
    alpha = tf.random.uniform([real_data.shape[0], 1, 1])
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolates)
        gp_output = discriminator(interpolates)
    gp_grad = gp_tape.gradient(gp_output, interpolates)
    norm = tf.norm(tf.reshape(gp_grad, [tf.shape(gp_grad)[0], -1]), axis=1)
    gp = lambda_gp * tf.reduce_mean((norm - 1.) ** 2)
    return gp

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2)

gen_losses = tf.keras.metrics.Mean(name='gen_losses')
disc_losses = tf.keras.metrics.Mean(name='disc_losses')
gp_losses = tf.keras.metrics.Mean(name='gp_losses')

# TODO: Modify the training_step as desired
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(real_data, gen_losses, disc_losses, gp_losses, n_critic, num_step):
    noise = tf.random.normal([real_data.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_samples = generator(noise, training=True)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_samples, training=True)

        gen_loss, disc_loss = wasserstein_gan_loss(real_output, fake_output)
        gp_loss = gradient_penalty(discriminator, real_data, generated_samples)
        total_disc_loss = disc_loss + gp_loss

    # generator training
    if num_step % n_critic == 0:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        gen_losses(gen_loss)
    # discriminator training
    else:
        gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        disc_losses(disc_loss)
        gp_losses(gp_loss)

def generate_samples(generator, filename, random_seeds):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode.
  samples = generator(random_seeds, training=False)
  # TODO : Modify scaling and refining method if you want
  samples = samples * 127

  # samples dtype should be integer and sample values should be in 0...127
  samples = tf.cast(samples, dtype=tf.int64)
  samples = samples.numpy()
  samples[samples < 0] = 0
  samples[samples > 127] = 127
  for i in range(samples.shape[0]):
      save_pianoroll_to_midi(samples[i], filename + '%d.midi' % i)

# TODO : Train your model and Save your best model
if not is_test_mode:
    # training Dataset Load
    train_loader = DataLoader(audio_dir=audio_dir, data_dir=data_dir)
    train_ds = train_loader.dataset()
    dataset = train_ds.shuffle(10000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    num_step = tf.zeros(1, dtype=tf.int64)
    for epoch in range(epochs):
        # Training
        for pianorolls in dataset:
            # TODO : Modify scaling method if you want
            pianorolls = pianorolls / 127

            train_step(pianorolls, gen_losses, disc_losses, gp_losses, n_critic, num_step)
            num_step += 1

        nn_structure = 'cnn_beta500'
        f = open(os.path.join(str(nn_structure) + '.log'), 'a')
        print('====== Epoch: {:03d}, Generator loss is {:.2f}, Discriminator loss is {:.2f} and Gradient penalty loss is {:.2f}'.format(epoch + 1,
                                                                                                                                        gen_losses.result(),
                                                                                                                                        disc_losses.result(),
                                                                                                                                        gp_losses.result()))
        f.write('====== Epoch: {:03d}, Generator loss is {:.2f}, Discriminator loss is {:.2f} and Gradient penalty loss is {:.2f}\n'.format(epoch + 1,
                                                                                                                                        gen_losses.result(),
                                                                                                                                        disc_losses.result(),
                                                                                                                                        gp_losses.result()))
        f.close()

        # Reset the metrics for the next epoch
        gen_losses.reset_states()
        disc_losses.reset_states()
        gp_losses.reset_states()

        # Generate samples to check the training result
        if not os.path.exists(os.path.join(sample_dir, nn_structure)):
            os.mkdir(os.path.join(sample_dir, nn_structure))
        generate_samples(generator, os.path.join(sample_dir, nn_structure, '%03d_epoch_' % (epoch + 1)), random_seeds)

        save_path = os.path.join(ckpt_dir, 'run-%03d-%02d%02d-%02d%02d' % (tuple([(epoch+1)]) + tuple(localtime(time()))[1:5]))
        generator.save_weights(save_path, save_format='tf')

        f = open(os.path.join(str(nn_structure) + '.log'), 'a')
        print('===== Generator model saved : %s' % save_path)
        f.write('===== Generator model saved : %s\n' % save_path)
        f.close()

# TODO : Do sampling
elif is_test_mode:
    # restore best model
    generator.load_weights(restore_path)
    print('===== Generator model restored : %s' % restore_path)

    # sampling and save
    generate_samples(generator, os.path.join(sample_dir, 'test_'), random_seeds)