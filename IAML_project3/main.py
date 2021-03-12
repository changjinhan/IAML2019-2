import os
import tensorflow as tf
from dataloader import DataLoader
from time import localtime, time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[2], True)

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
ckpt_dir = 'ckpt'
best_saved_model = 'run-%02d%02d-%02d%02d%.3f' % (11, 27, 23, 28, 0.627)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'music_dataset'
data_dir = 'data'

# TODO : Refer to other methods and code your signal process method (signal_process.py)
signal_process = 'your_own_way'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
epochs = 50
batch_size = 32
melody_length = 240
num_melody_label = 13
sr = 4096

# TODO : Build your model here
class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_3 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.conv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_4 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=[2, 1])

        self.conv_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_5 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.conv_6 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_6 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=[2, 1])

        self.conv_7 = tf.keras.layers.Conv2D(filters=128, kernel_size=[12, 9], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_7 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.conv_8 = tf.keras.layers.Conv2D(filters=13, kernel_size=[1, 1], padding='same', activation='linear',
                                             kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_8 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_3 = tf.keras.layers.AvgPool2D(pool_size=[20, 1])

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_melody_label))
        self.dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation(activation='softmax'))

    def call(self, x, training=False):
        x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        x = self.conv_1(x)
        x = self.batch_normalization_1(x, training=training)
        x = self.conv_2(x)
        x = self.batch_normalization_2(x, training=training)
        x = self.conv_3(x)
        x = self.batch_normalization_3(x, training=training)
        x = self.conv_4(x)
        x = self.batch_normalization_4(x, training=training)
        x = self.pool_1(x)
        x = self.dropout(x)
        # ---------------------------------------------------
        x = self.conv_5(x)
        x = self.batch_normalization_5(x, training=training)
        x = self.conv_6(x)
        x = self.batch_normalization_6(x, training=training)
        x = self.pool_2(x)
        x = self.dropout(x)
        # ---------------------------------------------------
        x = self.conv_7(x)
        x = self.batch_normalization_7(x, training=training)
        x = self.dropout(x)
        # ---------------------------------------------------
        x = self.conv_8(x)
        x = self.batch_normalization_8(x, training=training)
        x = self.pool_3(x)

        # ---------------------------------------------------
        x = self.dropout(x)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[x.shape[0], x.shape[1], -1])
        x = self.dense_1(x)
        logits = self.dense_2(x)

        return logits

model = YourModel()

# loss objective, optimizer and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
loss_metric = tf.keras.metrics.Mean(name='loss')
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(features, labels, model, loss_object, optimizer, loss_metric, accuracy_metric):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(loss)
    accuracy_metric(labels, predictions)

@tf.function
def validation_and_test_step(features, labels, model, loss_object, loss_metric, accuracy_metric):
    predictions = model(features, training=False)
    loss = loss_object(labels, predictions)
    loss_metric(loss)
    accuracy_metric(labels, predictions)

# TODO : Train your model and Save your best model
if not is_test_mode:
    # training Dataset Load
    train_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='training', signal_process=signal_process, sr=sr)
    train_ds = train_loader.dataset()
    dataset = train_ds.shuffle(10000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # validation Dataset Load
    valid_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='validation', signal_process=signal_process, sr=sr)
    valid_ds = valid_loader.dataset()
    dataset_val = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    file_tag = 'cnn'
    accuracy = 0
    for epoch in range(epochs):
        # Training
        for features, labels in dataset:
            train_step(features, labels, model, loss_object, optimizer, loss_metric, accuracy_metric)


        f = open(os.path.join(str(file_tag) + '_' + signal_process + '.log'), 'a')
        print('====== Epoch: {:03d}, Loss: {:.2f}, Accuracy: {:.3f} '.format(epoch + 1,
                                                                     loss_metric.result(),
                                                                     accuracy_metric.result()))
        f.write('====== Epoch: {:03d}, Loss: {:.2f}, Accuracy: {:.3f}\n'.format(epoch + 1,
                                                                        loss_metric.result(),
                                                                        accuracy_metric.result()))
        f.close()

        # Reset the metrics for the next epoch
        loss_metric.reset_states()
        accuracy_metric.reset_states()

        # Validation
        for features, labels in dataset_val:
            validation_and_test_step(features, labels, model, loss_object, loss_metric, accuracy_metric)

        val_accuracy = accuracy_metric.result()
        f = open(os.path.join(str(file_tag) + '_' + signal_process + '.log'), 'a')

        print('===== Validation, Loss: {:.2f}, Accuracy: {:.3f} '.format(loss_metric.result(),
                                                                  accuracy_metric.result()))
        f.write('===== Validation, Loss: {:.2f}, Accuracy: {:.3f}\n'.format(loss_metric.result(),
                                                                    accuracy_metric.result()))
        f.close()

        if accuracy < val_accuracy:
            accuracy = val_accuracy
            if not os.path.exists(os.path.join(ckpt_dir, str(file_tag))):
                os.mkdir(os.path.join(ckpt_dir, str(file_tag)))
            save_path = os.path.join(ckpt_dir, str(file_tag), 'run-%02d%02d-%02d%02d%.3f' % (tuple(localtime(time()))[1:5] + tuple([val_accuracy])))
            model.save_weights(save_path, save_format='tf')
            print('===== Model saved : %s; accuracy : %.3f' % (save_path, val_accuracy))

        # Reset the metrics for the next epoch
        loss_metric.reset_states()
        accuracy_metric.reset_states()

# TODO : Do accuracy test
elif is_test_mode:
    test_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='test', signal_process=signal_process, sr=sr)
    # test_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='validation', signal_process=signal_process, sr=sr)
    test_ds = test_loader.dataset()

    # restore best model
    model.load_weights(restore_path)
    print('===== Model restored : %s' % restore_path)

    # Model Test
    dataset = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for features, labels in dataset:
        validation_and_test_step(features, labels, model, loss_object, loss_metric, accuracy_metric)

    print('===== Test, Loss: {:.2f} Accuracy: {:.3f} '.format(loss_metric.result(),
                                                              accuracy_metric.result()))

