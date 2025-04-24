
import tensorflow as tf
from tensorflow.keras import layers

# You would need to define your LayerNormalization, PosEncoding, Linear, ScaledDotProductAttention exactly like you had them in PyTorch.

class _MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = layers.Dense(d_k * n_heads)
        self.w_k = layers.Dense(d_k * n_heads)
        self.w_v = layers.Dense(d_v * n_heads)

    def call(self, q, k, v):
        b_size = tf.shape(q)[0]
        q_s = tf.transpose(tf.reshape(self.w_q(q), [b_size, -1, self.n_heads, self.d_k]), perm=[0, 2, 1, 3])
        k_s = tf.transpose(tf.reshape(self.w_k(k), [b_size, -1, self.n_heads, self.d_k]), perm=[0, 2, 1, 3])
        v_s = tf.transpose(tf.reshape(self.w_v(v), [b_size, -1, self.n_heads, self.d_v]), perm=[0, 2, 1, 3])
        return q_s, k_s, v_s

class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = layers.Conv1D(d_ff, 1)
        self.conv2 = layers.Conv1D(d_model, 1)
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.relu = layers.ReLU()

    def call(self, inputs):
        residual = inputs
        output = self.conv1(tf.transpose(inputs, perm=[0,2,1]))
        output = self.relu(output)
        output = self.conv2(output)
        output = tf.transpose(output, perm=[0,2,1])
        output = self.dropout(output)
        return self.layer_norm(residual + output)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = layers.Dense(d_model)
        self.linear_s = layers.Dense(d_model)
        self.proj_v = layers.Dense(d_model)
        self.proj_s = layers.Dense(d_model)
        self.d_v = d_v
        self.dropout = layers.Dropout(dropout)
        self.layer_norm_v = LayerNormalization(d_model)
        self.layer_norm_s = LayerNormalization(d_model)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def call(self, v, s, v_len, s_len):
        b_size = tf.shape(v)[0]
        v = self.linear_v(v)
        s = self.linear_s(s)

        if self.pos:
            pos_v = self.pos_emb_v(v_len)
            pos_s = self.pos_emb_s(s_len)
            residual_v = v + pos_v
            residual_s = s + pos_s
        else:
            residual_v = v
            residual_s = s

        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)

        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)

        context_v = tf.reshape(tf.transpose(context_v, perm=[0,2,1,3]), [b_size, -1, self.n_heads * self.d_v])
        context_s = tf.reshape(tf.transpose(context_s, perm=[0,2,1,3]), [b_size, -1, self.n_heads * self.d_v])

        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))

        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)

class CoAttention(tf.keras.Model):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(CoAttention, self).__init__()
        self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
                                             visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos)
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)

    def call(self, v, s, v_len, s_len):
        v, s = self.multi_head(v, s, v_len, s_len)
        v = self.PoswiseFeedForwardNet_v(v)
        s = self.PoswiseFeedForwardNet_s(s)
        return v, s
