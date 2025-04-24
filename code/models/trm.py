
import numpy as np
import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = tf.keras.layers.Dense(out_features, use_bias=bias,
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            bias_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs):
        return self.linear(inputs)

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = tf.math.sqrt(tf.cast(d_k, tf.float32))
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, q, k, v, attn_mask=None):
        scores = tf.matmul(q, k, transpose_b=True) / self.scale_factor
        if attn_mask is not None:
            scores = tf.where(attn_mask, tf.ones_like(scores) * (-1e9), scores)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        context = tf.matmul(attn, v)
        return context, attn

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = self.add_weight(shape=(d_hid,), initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=(d_hid,), initializer="zeros", trainable=True)
        self.eps = eps

    def call(self, z):
        mean = tf.reduce_mean(z, axis=-1, keepdims=True)
        std = tf.math.reduce_std(z, axis=-1, keepdims=True)
        ln_out = (z - mean) / (std + self.eps)
        return self.gamma * ln_out + self.beta

class PosEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
             for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        self.pos_enc = tf.keras.layers.Embedding(input_dim=max_seq_len + 1,
                                                 output_dim=d_word_vec,
                                                 embeddings_initializer=tf.constant_initializer(pos_enc),
                                                 trainable=False)
        self.max_len = int(max_seq_len / 10)

    def call(self, input_len):
        batch_size = tf.shape(input_len)[0]
        input_pos = []
        for l in input_len:
            seq = tf.range(1, l + 1)
            pad = tf.zeros([self.max_len - l], dtype=tf.int32)
            input_pos.append(tf.concat([seq, pad], axis=0))
        input_pos = tf.stack(input_pos)
        return self.pos_enc(input_pos)
