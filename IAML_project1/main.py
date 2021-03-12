import os
import tensorflow as tf
from dataloader import DataLoader
from time import localtime, time

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
ckpt_dir = 'ckpt'
best_saved_model = 'run-%02d%02d-%02d%02d%.3f' % (10, 28, 4, 12, 0.555)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'music_dataset'
data_dir = 'data'

# TODO : Utilize your signal process method (signal_process.py)
signal_process = 'your_own_way'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
epochs = 1000
batch_size = 32
num_label = 8
learning_rate = 1e-3


# TODO : Build your model here
class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()
        # cqt
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization1 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=[2, 4], strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization2 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=[2, 4], strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization3 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_3 = tf.keras.layers.MaxPool2D(pool_size=[2, 4], strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization4 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 3], strides=2, padding='same')
        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization5 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=[4, 6], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization6 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        self.pool_6 = tf.keras.layers.MaxPool2D(pool_size=[4, 6], strides=2, padding='same')
        self.conv7 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization7 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True, trainable=True)
        # ----

        # mel 1
        self.conv_m_1_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[50, 3], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_1_1 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_1_1 = tf.keras.layers.MaxPool2D(pool_size=[10, 4], strides=2, padding='same')
        self.conv_m_1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=[50, 3], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_1_2 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_1_2 = tf.keras.layers.MaxPool2D(pool_size=[10, 3], strides=2, padding='same')
        self.conv_m_1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_1_3 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_1_3 = tf.keras.layers.MaxPool2D(pool_size=[5, 5], strides=2, padding='same')
        self.conv_m_1_4 = tf.keras.layers.Conv2D(filters=16, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_1_4 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        # self.conv_m_1_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[50, 2], padding='same', activation=tf.nn.relu,
        #                                          kernel_initializer=tf.keras.initializers.he_normal())
        # self.batch_normalization_m_1_4 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
        #                                                                     trainable=True)
        # self.pool_m_1_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 3], strides=2, padding='same')
        # self.conv_m_1_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=[50, 2], padding='same',
        #                                          activation=tf.nn.relu,
        #                                          kernel_initializer=tf.keras.initializers.he_normal())
        # self.batch_normalization_m_1_5 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
        #                                                                     trainable=True)
        # self.pool_m_1_5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        # self.conv_m_1_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=[50, 2], padding='same',
        #                                          activation=tf.nn.relu,
        #                                          kernel_initializer=tf.keras.initializers.he_normal())
        # self.batch_normalization_m_1_6 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
        #                                                                     trainable=True)
        # self.pool_m_1_6 = tf.keras.layers.MaxPool2D(pool_size=[4, 6], strides=2, padding='same')
        # self.conv_m_1_7 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
        #                                          activation=tf.nn.relu,
        #                                          kernel_initializer=tf.keras.initializers.he_normal())
        # self.batch_normalization_m_1_7 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
        #                                                                     trainable=True)
        # ----

        # mel 2
        self.conv_m_2_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_1 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_1 = tf.keras.layers.MaxPool2D(pool_size=[2, 6], strides=2, padding='same')
        self.conv_m_2_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_2 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_2 = tf.keras.layers.MaxPool2D(pool_size=[2, 6], strides=2, padding='same')
        self.conv_m_2_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_3 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_3 = tf.keras.layers.MaxPool2D(pool_size=[2, 5], strides=2, padding='same')
        self.conv_m_2_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_4 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 5], strides=2, padding='same')
        self.conv_m_2_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 5], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_5 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_5 = tf.keras.layers.MaxPool2D(pool_size=[2, 4], strides=2, padding='same')
        self.conv_m_2_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=[4, 4], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_6 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        self.pool_m_2_6 = tf.keras.layers.MaxPool2D(pool_size=[4, 4], strides=2, padding='same')
        self.conv_m_2_7 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                                 kernel_initializer=tf.keras.initializers.he_normal())
        self.batch_normalization_m_2_7 = tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True,
                                                                            trainable=True)
        # ----

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(num_label, activation='softmax')

    def call(self, x, training=False):
        first_shape = 84
        x1 = x[:, :first_shape, :]
        x2 = x[:, first_shape:, :]
        x1 = tf.reshape(x1, (x1.shape[0], x1.shape[1], x1.shape[2], 1))
        x2 = tf.reshape(x2, (x2.shape[0], x2.shape[1], x2.shape[2], 1))

        # cqt
        x1 = self.conv1(x1)
        x1 = self.batch_normalization1(x1, training=training)
        x1 = self.pool_1(x1)
        x1 = self.conv2(x1)
        x1 = self.batch_normalization2(x1, training=training)
        x1 = self.pool_2(x1)
        x1 = self.conv3(x1)
        x1 = self.batch_normalization3(x1, training=training)
        x1 = self.pool_3(x1)
        x1 = self.conv4(x1)
        x1 = self.batch_normalization4(x1, training=training)
        x1 = self.pool_4(x1)
        x1 = self.conv5(x1)
        x1 = self.batch_normalization5(x1, training=training)
        x1 = self.pool_5(x1)
        x1 = self.conv6(x1)
        x1 = self.batch_normalization6(x1, training=training)
        x1 = self.pool_6(x1)
        x1 = self.conv7(x1)
        x1 = self.batch_normalization7(x1, training=training)
        # ----

        # mel 1
        x2_1 = self.conv_m_1_1(x2)
        x2_1 = self.batch_normalization_m_1_1(x2_1, training=training)
        x2_1 = self.pool_m_1_1(x2_1)
        x2_1 = self.conv_m_1_2(x2_1)
        x2_1 = self.batch_normalization_m_1_2(x2_1, training=training)
        x2_1 = self.pool_m_1_2(x2_1)
        x2_1 = self.conv_m_1_3(x2_1)
        x2_1 = self.batch_normalization_m_1_3(x2_1, training=training)
        x2_1 = self.pool_m_1_3(x2_1)
        x2_1 = self.conv_m_1_4(x2_1)
        x2_1 = self.batch_normalization_m_1_4(x2_1, training=training)
        # x2_1 = self.pool_m_1_4(x2_1)
        # x2_1 = self.conv_m_1_5(x2_1)
        # x2_1 = self.batch_normalization_m_1_5(x2_1, training=training)
        # x2_1 = self.pool_m_1_5(x2_1)
        # x2_1 = self.conv_m_1_6(x2_1)
        # x2_1 = self.batch_normalization_m_1_6(x2_1, training=training)
        # x2_1 = self.pool_m_1_6(x2_1)
        # x2_1 = self.conv_m_1_7(x2_1)
        # x2_1 = self.batch_normalization_m_1_7(x2_1, training=training)
        # ----

        # mel 2
        x2_2 = self.conv_m_2_1(x2)
        x2_2 = self.batch_normalization_m_2_1(x2_2, training=training)
        x2_2 = self.pool_m_2_1(x2_2)
        x2_2 = self.conv_m_2_2(x2_2)
        x2_2 = self.batch_normalization_m_2_2(x2_2, training=training)
        x2_2 = self.pool_m_2_2(x2_2)
        x2_2 = self.conv_m_2_3(x2_2)
        x2_2 = self.batch_normalization_m_2_3(x2_2, training=training)
        x2_2 = self.pool_m_2_3(x2_2)
        x2_2 = self.conv_m_2_4(x2_2)
        x2_2 = self.batch_normalization_m_2_4(x2_2, training=training)
        x2_2 = self.pool_m_2_4(x2_2)
        x2_2 = self.conv_m_2_5(x2_2)
        x2_2 = self.batch_normalization_m_2_5(x2_2, training=training)
        x2_2 = self.pool_m_2_5(x2_2)
        x2_2 = self.conv_m_2_6(x2_2)
        x2_2 = self.batch_normalization_m_2_6(x2_2, training=training)
        x2_2 = self.pool_m_2_6(x2_2)
        x2_2 = self.conv_m_2_7(x2_2)
        x2_2 = self.batch_normalization_m_2_7(x2_2, training=training)

        # ----

        x1 = self.flatten(x1)
        x2_1 = self.flatten(x2_1)
        x2_2 = self.flatten(x2_2)
        x = tf.concat([x1, x2_1, x2_2], 1)
        x = self.dropout(x, training=training)
        return self.dense(x)


model = YourModel()

# loss objective, optimizer and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
    print('start train loader')
    train_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='training',
                              signal_process=signal_process)
    print('\ttrain loader end')
    print('start train data set')
    train_ds = train_loader.dataset()
    print('\ttrain data set end')
    print('start shuffle')
    buffer_size = 7000
    train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    print('\tshuffle end')
    accuracy = 0
    print('start train')
    # training Dataset Load
    for epoch in range(epochs):
        print('epoch :', epoch)
        # Training
        print('\ttrain start')
        temp_a = 0
        for features, labels in train_ds:
            print('\t\t\t', '%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5], temp_a)
            train_step(features=features, labels=labels, model=model, loss_object=loss_object, optimizer=optimizer,
                       loss_metric=loss_metric, accuracy_metric=accuracy_metric)
            temp_a += 1
        print('\t\ttrain end')

        print('\tget results start')
        result_loss = loss_metric.result()
        result_accuracy = accuracy_metric.result()
        print('\t\tget results end')

        print('\trecording start')
        f = open(os.path.join(str(buffer_size), signal_process + '.log'), 'a')
        print('====== Epoch: {:03d}, Loss: {:.2f} Accuracy: {:.3f} '.format(epoch + 1, result_loss, result_accuracy))
        f.write('====== Epoch: {:03d}, Loss: {:.2f} Accuracy: {:.3f} \n'.format(epoch + 1, result_loss, result_accuracy))
        f.close()
        print('\t\trecording end')
        # Reset the metrics for the next epoch
        print('\tinitializing start')
        loss_metric.reset_states()
        accuracy_metric.reset_states()
        print('\t\tinitializing end')

        print('\tvalidation start')
        # validation Dataset Load
        valid_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir,
                                  split='validation', signal_process=signal_process)
        valid_ds = valid_loader.dataset()
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Validation
        temp_a = 0
        for features, labels in valid_ds:
            print('\t\t\t', localtime(time()), temp_a)
            validation_and_test_step(features, labels, model, loss_object, loss_metric, accuracy_metric)
            temp_a += 1

        print('\t\tvalidation end')
        val_loss = loss_metric.result()
        val_accuracy = accuracy_metric.result()
        f = open(os.path.join(str(buffer_size), signal_process + '.log'), 'a')
        print('===== Validation, Loss: {:.2f} Accuracy: {:.3f} '.format(val_loss, val_accuracy))
        f.write('===== Validation, Loss: {:.2f} Accuracy: {:.3f} \n'.format(val_loss, val_accuracy))
        f.close()

        if accuracy < val_accuracy:
            accuracy = val_accuracy
            save_path = os.path.join(str(buffer_size), ckpt_dir, 'run-%02d%02d-%02d%02d%.3f' % (
                        tuple(localtime(time()))[1:5] + tuple([val_accuracy])))
            if not os.path.exists(os.path.join(str(buffer_size), ckpt_dir)):
                os.mkdir(os.path.join(str(buffer_size), ckpt_dir))
            model.save_weights(save_path, save_format='tf')
            print('===== Model saved : %s; Accuracy : %.3f' % (save_path, val_accuracy))
            model.summary()
            print()

        # Reset the metrics for the next epoch
        loss_metric.reset_states()
        accuracy_metric.reset_states()


# TODO : Do accuracy test
if is_test_mode:
    test_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='test',
                             signal_process=signal_process)
    test_ds = test_loader.dataset()
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # restore best model
    model.load_weights(restore_path)
    print('===== Model restored : %s' % restore_path)

    # Model Test
    for features, labels in test_ds:
        validation_and_test_step(features, labels, model, loss_object, loss_metric, accuracy_metric)

    print('===== Test, Loss: {:.2f} Accuracy: {:.3f} '.format(loss_metric.result(),
                                                              accuracy_metric.result()))
