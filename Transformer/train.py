import sys
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Transformer.model import TransformerClassifier, create_padding_mask
from metrics import micro_f1, macro_f1

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from TextCNN.train import load_testcnn_data
from config import *


def load_testcnn_data(x_npy, y_npy):
    x = np.load(x_npy).astype(np.float32)
    y = np.load(y_npy).astype(np.float32)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2020)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2020)
    return train_x, test_x, dev_x, train_y, test_y, dev_y


def predict(model, x, batch_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    res = []
    for batch_x in dataset:
        enc_padding_mask = create_padding_mask(batch_x)
        y_pred = model(batch_x, training=False, enc_padding_mask=enc_padding_mask)
        res.append(y_pred)
    res = tf.concat(res, axis=0)
    return res

def evaluation(x, y):
    y = tf.cast(y, dtype=tf.float32)
    y_pred = predict(x)

    predict_accuracy = tf.keras.metrics.BinaryAccuracy(name='predict_accuracy')
    acc = predict_accuracy(y, y_pred)
    mi_f1 = micro_f1(y, y_pred)
    ma_f1 = macro_f1(y, y_pred)
    print("val accuracy {:.4f}, micro f1 {:.4f} macro f1 {:.4f}".format(
        acc.numpy(), mi_f1.numpy(), ma_f1.numpy()))
    return acc, mi_f1, ma_f1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    config = config()
    train_x, test_x, dev_x, train_y, test_y, dev_y = load_testcnn_data(config.X_NPY_PATH, config.Y_NPY_PATH)

    tokenizer, mlb = load_tokenizer_binarizer(config.TOKENIZER_BINARIZER)
    BUFFER_SIZE = 256
    BATCH_SIZE = 128

    train_dataset = tf.data.Dataset.from_tensor_slices(train_x, train_y)
    dev_dataset = tf.data.Dataset.from_tensor_slices(dev_x, dev_y)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_x, test_y)

    dev_dataset = dev_dataset.cache()
    train_dataset = train_dataset.cache()

    train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE,
                                                                                            drop_remainder=True)

    dev_dataset = dev_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    vocab_size = 50000
    maximum_position_encoding = 10000
    output_dim = len(train_y[0])
    dropout_rate = 0.1

    steps_per_epoch = len(train_x)
    epochs = 5

    ckpt_path = '../data/Transformer/train'

    model = TransformerClassifier(num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding,
                                  output_dim, dropout_rate)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='auto')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest ckpt restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(x, y):
        enc_padding_mask = create_padding_mask(x)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True, enc_padding_mask=enc_padding_mask)
            loss = loss_object(y, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y, y_pred)

        mi_f1 = micro_f1(y, y_pred)
        ma_f1 = macro_f1(y, y_pred)
        return mi_f1, ma_f1, y_pred


    for epoch in range(epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (x, y) in enumerate(train_dataset.take(steps_per_epoch)):
            mi_f1, ma_f1, _ = train_step(x, y)
            if batch % 20 == 0:
                print('epoch:{} batch:{:.3d} loss:{:4f} acc:{:.4f} micro_f1:{:.4f} macro_f1:{:.4f}'.format(epoch + 1, batch + 1, train_loss.result(), train_accuracy.result(), mi_f1, ma_f1))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving ckpt for epoch {} to {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} loss {}'.format(epoch + 1, train_loss.result()))
        evaluation(dev_x, dev_y)
        print('Time taken 1 epoch {:.2f} sec\n'.format(time.time() - start))

