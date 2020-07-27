import tensorflow as tf
import time
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    position_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(position_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  # =0时 为1， 否则为1
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# encoder中的mask是padding； decoder中的mask是look_ahead
def creat_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    q:(..., seq_len_q, depth)
    k:(..., seq_len_k, depth)
    v:(..., seq_len_v, depth)
    mask:(...,seql_len_q, seql_len_k)

    输出O.shape:(..., seq_len_q, depth)
    注意力权重 shape:(..., seq_len_q, sel_lev_v)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, seq_len_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def softmax(x):
    """Compute the softmax in a numerically stable way."""  # 防止如果x中的值太大，会溢出
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


if __name__ == '__main__':
    x = np.array([1, 2, 3])
    mask = np.array([0, 0, 1])
    print(mask)
    print(mask * -1e9)
    print(-1e9)
    print(1e3)
    if 1e3 == 1000:
        print('true')
    else:
        print('false')
