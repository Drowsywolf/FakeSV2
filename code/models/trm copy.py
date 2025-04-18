import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np


class Linear(layers.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = layers.Dense(
            units=out_features,
            use_bias=bias,
            kernel_initializer=initializers.GlorotNormal(), 
            bias_initializer='zeros'
        )

    def call(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(layers.Layer):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale_factor = tf.math.sqrt(tf.cast(d_k, tf.float32))
        self.softmax = tf.nn.softmax
        self.dropout = layers.Dropout(dropout)

    def call(self, q, k, v, attn_mask=None, training=False):

        scores = tf.matmul(q, k, transpose_b=True) / self.scale_factor
        

        if attn_mask is not None:
            scores = tf.where(attn_mask, -1e9, scores)
        
        attn = self.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)
        

        context = tf.matmul(attn, v)
        return context, attn


class LayerNormalization(layers.Layer):
    def __init__(self, d_hid, eps=1e-6):
        super().__init__()
        self.gamma = self.add_weight(
            shape=(d_hid,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            shape=(d_hid,),
            initializer='zeros',
            trainable=True
        )
        self.eps = eps

    def call(self, z):
        mean = tf.reduce_mean(z, axis=-1, keepdims=True)
        std = tf.math.reduce_std(z, axis=-1, keepdims=True)
        ln_out = (z - mean) / (std + self.eps)
        return self.gamma * ln_out + self.beta


class PosEncoding(layers.Layer):
    def __init__(self, max_seq_len, d_word_vec):
        super().__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)]
        )
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)


        self.pos_enc = layers.Embedding(
            input_dim=max_seq_len + 1,
            output_dim=d_word_vec,
            weights=[pos_enc],
            trainable=False
        )
        self.max_len = int(max_seq_len / 10)

    def call(self, input_len):
        batch_size = tf.shape(input_len)[0]
        positions = tf.stack([
            tf.concat([tf.range(1, len+1), tf.zeros(self.max_len-len, dtype=tf.int32)], 0)
            for len in input_len
        ])
        return self.pos_enc(positions)