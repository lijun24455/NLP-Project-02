import sys
import os

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import time

from sklearn.model_selection import train_test_split
from TextCNN.model import TextCNN
from config import *
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, classification_report


def load_testcnn_data(x_npy, y_npy):
    x = np.load(x_npy).astype(np.float32)
    y = np.load(y_npy).astype(np.float32)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2020)
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    config = config()
    train_x, test_x, train_y, test_y = load_testcnn_data(config.X_NPY_PATH, config.Y_NPY_PATH)

    # print('train_x.shape:', train_x.shape)  # 23850,200
    # print('train_y.shape:', train_y.shape)  # 23850,942

    args = config.getArgs()
    args.output_dim = len(train_y[0])
    args.steps_per_epoch = len(train_y) // args.batch_size
    kernel_sizes = [int(i) for i in args.kernel_sizes.split(',')]

    print(args.checkpoint_path)
    print('output_dim:', args.output_dim, 'kernel_sizes:', kernel_sizes)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(256)
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)

    model = TextCNN(max_len=args.max_len, vocab_size=args.vocab_size, embedding_dim=args.embedding_dim,
                    output_dim=args.output_dim, kernel_sizes=kernel_sizes)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss_metric')

    # checkpoint
    ckpt = tf.train.Checkpoint(textcnn=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_path, max_to_keep=1)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')


    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_object(y, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        return y_pred


    for epoch in range(args.epochs):
        start = time.time()
        train_loss.reset_states()

        for batch_cnt, (x, y) in enumerate(train_dataset.take(args.steps_per_epoch)):
            y_pred = train_step(x, y)
            if batch_cnt % 20 == 0:
                print('epoch {} batch {} loss {}'.format(epoch + 1, batch_cnt + 1, train_loss.result()))
        ckpt_save_path = ckpt_manager.save()
        print('Saving ckpt for epoch {} to {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {}'.format(epoch + 1, train_loss.result()))

        print('Epoch {} Time Cost:{}'.format(epoch + 1, time.time() - start))

    y_pred = model(test_x)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    tokenizer, mlb = load_tokenizer_binarizer(config.TOKENIZER_BINARIZER)

    print('y-pred:', mlb.inverse_transform(y_pred[:10]))
    print('y-real:', mlb.inverse_transform(test_y[:10]))

    print(classification_report(test_y, y_pred))