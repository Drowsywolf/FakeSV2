import tensorflow as tf
import numpy as np

# No exact equivalent of torch.autograd.Function needed here for ReverseLayerF — we implement it manually.

class ReverseLayerF(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0):
        super(ReverseLayerF, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -self.alpha * dy
            return x, grad

        return reverse(inputs)

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads=2, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = tf.keras.layers.Softmax(axis=-1)
        self.to_qkv = tf.keras.layers.Dense(inner_dim * 3, use_bias=False)

        if project_out:
            self.to_out = tf.keras.Sequential([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.Dropout(dropout)
            ])
        else:
            self.to_out = tf.identity

    def call(self, x):
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = [tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], self.heads, -1)) for t in qkv]
        q, k, v = [tf.transpose(t, perm=[0, 2, 1, 3]) for t in [q, k, v]]

        dots = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = self.attend(dots)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], -1))

        return self.to_out(out)
