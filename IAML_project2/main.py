import os
import tensorflow as tf
from dataloader import DataLoader
from time import localtime, time

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
ckpt_dir = 'ckpt'
best_saved_model = 'run-1110-10380.7050.885'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'music_dataset'
data_dir = 'data'

# TODO : Refer to other methods and code your signal process method (signal_process.py)
signal_process = 'tf_log_mel_spectrogram'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
epochs = 100
batch_size = 32
label_names = ['happy', 'film', 'energetic', 'relaxing', 'emotional', 'melodic', 'dark', 'epic', 'dream', 'love']

# TODO : Build your model here
class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()
        # mel 1 CNN
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
        # mel 2 CNN
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
        # Bidirectional LSTM
        self.bi_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
        self.bi_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
        # self.bi_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))


        # ——

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        # mel 1
        x1 = self.conv_m_1_1(x)
        x1 = self.batch_normalization_m_1_1(x1, training=training)
        x1 = self.pool_m_1_1(x1)
        x1 = self.conv_m_1_2(x1)
        x1 = self.batch_normalization_m_1_2(x1, training=training)
        x1 = self.pool_m_1_2(x1)
        x1 = self.conv_m_1_3(x1)
        x1 = self.batch_normalization_m_1_3(x1, training=training)
        x1 = self.pool_m_1_3(x1)
        x1 = self.conv_m_1_4(x1)
        x1 = self.batch_normalization_m_1_4(x1, training=training)

        x1 = self.flatten(x1)
        x1 = tf.reshape(x1, [x1.shape[0],x1.shape[1],1])
        x1 = self.bi_lstm1(x1)
        x1 = self.bi_lstm2(x1)
        x1 = x1[:,-1,:]
        print(x1.shape)



        # mel 2
        x2 = self.conv_m_2_1(x)
        x2 = self.batch_normalization_m_2_1(x2, training=training)
        x2 = self.pool_m_2_1(x2)
        x2 = self.conv_m_2_2(x2)
        x2 = self.batch_normalization_m_2_2(x2, training=training)
        x2 = self.pool_m_2_2(x2)
        x2 = self.conv_m_2_3(x2)
        x2 = self.batch_normalization_m_2_3(x2, training=training)
        x2 = self.pool_m_2_3(x2)
        x2 = self.conv_m_2_4(x2)
        x2 = self.batch_normalization_m_2_4(x2, training=training)
        x2 = self.pool_m_2_4(x2)
        x2 = self.conv_m_2_5(x2)
        x2 = self.batch_normalization_m_2_5(x2, training=training)
        x2 = self.pool_m_2_5(x2)
        x2 = self.conv_m_2_6(x2)
        x2 = self.batch_normalization_m_2_6(x2, training=training)
        x2 = self.pool_m_2_6(x2)
        x2 = self.conv_m_2_7(x2)
        x2 = self.batch_normalization_m_2_7(x2, training=training)
        # ——

        x2= self.flatten(x2)
        x2 = tf.reshape(x2, [x2.shape[0],x2.shape[1],1])
        x2 = self.bi_lstm1(x2)
        x2 = self.bi_lstm2(x2)
        x2 = x2[:,-1,:]

        x = tf.concat([x1,x2], axis=1)

        x = self.dropout(x, training=training)
        print(self.dense(x).shape)
        return self.dense(x)

model = YourModel()

# loss objective, optimizer and metrics
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
loss_metric = tf.keras.metrics.Mean(name='loss')
# accuracy metric is not for scoring of project 2, just reference
accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')

# ROC (Receiver Operation Characteristic Curve) - AUC (Area Under the Curve) metric
roc_auc_dict = dict()
for label in label_names:
    # TODO: Do not change the arguments of tf.keras.metrics.AUC function
    roc_auc_dict[label] = tf.keras.metrics.AUC()

# mean ROC-AUC from roc_auc_dict
def mean_roc_auc(roc_auc_dict, label_names):
    num_label = len(label_names)
    sum = tf.zeros(1)
    for label in label_names:
        sum += roc_auc_dict[label].result()
    auc_score = sum / num_label
    return auc_score[0]

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(features, labels, model, loss_object, optimizer, loss_metric, roc_auc_dict, accuracy_metric, label_names):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(loss)
    accuracy_metric(labels, predictions)
    for i in range(len(label_names)):
        roc_auc_dict[label_names[i]](labels[:, i], predictions[:, i])

@tf.function
def validation_and_test_step(features, labels, model, loss_object, loss_metric, roc_auc_dict, accuracy_metric, label_names):
    predictions = model(features, training=False)
    loss = loss_object(labels, predictions)
    loss_metric(loss)
    accuracy_metric(labels, predictions)
    for i in range(len(label_names)):
        roc_auc_dict[label_names[i]](labels[:, i], predictions[:, i])

# TODO : Train your model and Save your best model
if not is_test_mode:
    # training Dataset Load
    train_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='training', signal_process=signal_process)
    train_ds = train_loader.dataset()
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    accuracy = 0
    roc_auc = 0
    for epoch in range(epochs):
        # Training
        for features, labels in train_ds:
            train_step(features, labels, model, loss_object, optimizer, loss_metric, roc_auc_dict, accuracy_metric, label_names)

        f = open(os.path.join(signal_process + '.log'), 'a')

        print('====== Epoch: {:03d}, Loss: {:.2f}, Mean ROC-AUC score for 10 labels: {:.3f}, accuracy {:.3f}'.format(epoch + 1,
                                                                                                                     loss_metric.result(),
                                                                                                                     mean_roc_auc(roc_auc_dict, label_names),
                                                                                                                     accuracy_metric.result()))
        f.write('====== Epoch: {:03d}, Loss: {:.2f}, Mean ROC-AUC score for 10 labels: {:.3f}, accuracy {:.3f}\n'.format(epoch + 1,
                                                                                                                     loss_metric.result(),
                                                                                                                     mean_roc_auc(roc_auc_dict, label_names),
                                                                                                                     accuracy_metric.result()))
        f.close()

        # Reset the metrics for the next epoch
        loss_metric.reset_states()
        accuracy_metric.reset_states()
        for label in label_names:
            roc_auc_dict[label].reset_states()

        # validation Dataset Load
        valid_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir,
                                  split='validation', signal_process=signal_process)
        valid_ds = valid_loader.dataset()
        valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Validation
        for features, labels in valid_ds:
            validation_and_test_step(features, labels, model, loss_object, loss_metric, roc_auc_dict,
                                     accuracy_metric, label_names)
        f = open(os.path.join(signal_process + '.log'), 'a')

        print('===== Validation, Epoch: {:03d}, Loss: {:.2f}, Mean ROC-AUC score for 10 labels: {:.3f}, accuracy {:.3f}'.format(
            epoch+1,
            loss_metric.result(),
            mean_roc_auc(roc_auc_dict, label_names),
            accuracy_metric.result()))

        f.write('===== Validation, Epoch: {:03d}, Loss: {:.2f}, Mean ROC-AUC score for 10 labels: {:.3f}, accuracy {:.3f}\n'.format(
            epoch+1,
            loss_metric.result(),
            mean_roc_auc(roc_auc_dict, label_names),
            accuracy_metric.result()))

        val_accuracy = accuracy_metric.result()
        val_mean_roc_auc = mean_roc_auc(roc_auc_dict, label_names)

        f.close()

        if roc_auc < val_mean_roc_auc:
            roc_auc = val_mean_roc_auc
            save_path = os.path.join(ckpt_dir, 'run-%02d%02d-%02d%02d%.3f%.3f' % (
                        tuple(localtime(time()))[1:5] + tuple([val_mean_roc_auc])+tuple([val_accuracy])))
            if not os.path.exists(os.path.join(ckpt_dir)):
                os.mkdir(os.path.join(ckpt_dir))
            model.save_weights(save_path, save_format='tf')
            print('===== Model saved : %s; mean_roc_auc : %.3f; accuracy : %.3f' % (save_path, val_mean_roc_auc, val_accuracy))
            model.summary()
            print()

        loss_metric.reset_states()
        accuracy_metric.reset_states()
        for label in label_names:
            roc_auc_dict[label].reset_states()

    # save_path = os.path.join(ckpt_dir,'run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5])
    # model.save_weights(save_path, save_format='tf')
    # print('===== Model saved : %s' % save_path)



# TODO : Do ROC-AUC test
elif is_test_mode:
    test_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='test', signal_process=signal_process)
    # test_loader = DataLoader(metadata_path=metadata_path, audio_dir=audio_dir, data_dir=data_dir, split='training', signal_process=signal_process)

    test_ds = test_loader.dataset()
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # restore best model
    model.load_weights(restore_path)
    print('===== Model restored : %s' % restore_path)

    # Model Test
    label_accuracy = dict()
    for features, labels in test_ds:
        validation_and_test_step(features, labels, model, loss_object, loss_metric, roc_auc_dict, accuracy_metric, label_names)
        predictions = model(features, training=False)
    print('===== Test, Loss: {:.2f}, Mean ROC-AUC score for 10 labels: {:.3f}, accuracy {:.3f}'.format(loss_metric.result(),
                                                                                                       mean_roc_auc(roc_auc_dict, label_names),
                                                                                                       accuracy_metric.result()))



