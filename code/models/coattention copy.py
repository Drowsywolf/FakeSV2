import tensorflow as tf
from tensorflow.keras import layers, Model

class _MultiHeadAttention(layers.Layer):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        

        self.w_q = layers.Dense(d_k * n_heads)
        self.w_k = layers.Dense(d_k * n_heads)
        self.w_v = layers.Dense(d_v * n_heads)

    def call(self, q, k, v):

        b_size = tf.shape(q)[0]
        

        q_s = self.w_q(q)  # (b, seq, d_k*n_heads)
        q_s = tf.reshape(q_s, [b_size, -1, self.n_heads, self.d_k])
        q_s = tf.transpose(q_s, [0, 2, 1, 3])  # (b, heads, seq, d_k)
        
        k_s = self.w_k(k)
        k_s = tf.reshape(k_s, [b_size, -1, self.n_heads, self.d_k])
        k_s = tf.transpose(k_s, [0, 2, 1, 3])
        
        v_s = self.w_v(v)
        v_s = tf.reshape(v_s, [b_size, -1, self.n_heads, self.d_v])
        v_s = tf.transpose(v_s, [0, 2, 1, 3])
        
        return q_s, k_s, v_s

class PoswiseFeedForwardNet(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.conv1 = layers.Conv1D(d_ff, 1, activation='relu')
        self.conv2 = layers.Conv1D(d_model, 1)
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=False):

        residual = inputs  # (b, seq, d_model)
        output = self.conv1(tf.transpose(inputs, [0, 2, 1]))  # (b, d_model, seq) => (b, d_ff, seq)
        output = tf.transpose(output, [0, 2, 1])  # (b, seq, d_ff)
        output = self.conv2(tf.transpose(output, [0, 2, 1]))  # (b, d_ff, seq) => (b, d_model, seq)
        output = tf.transpose(output, [0, 2, 1])  # (b, seq, d_model)
        output = self.dropout(output, training=training)
        return self.layer_norm(residual + output)

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super().__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)

        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = layers.Dense(d_model)
        self.linear_s = layers.Dense(d_model)
        self.proj_v = layers.Dense(d_model)
        self.proj_s = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
        self.layer_norm_v = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_s = layers.LayerNormalization(epsilon=1e-6)

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def call(self, v, s, v_len, s_len, training=False):
        b_size = tf.shape(v)[0]
        v = self.linear_v(v)  # (b, seq_v, d_model)
        s = self.linear_s(s)  # (b, seq_s, d_model)
        
        if self.pos:
            pos_v = self.pos_emb_v(v_len)
            pos_s = self.pos_emb_s(s_len)
            residual_v = v + pos_v
            residual_s = s + pos_s
        else:
            residual_v, residual_s = v, s
        

        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        

        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)
        

        context_v = tf.transpose(context_v, [0, 2, 1, 3])
        context_v = tf.reshape(context_v, [b_size, -1, self.n_heads * self.d_v])
        
        context_s = tf.transpose(context_s, [0, 2, 1, 3])
        context_s = tf.reshape(context_s, [b_size, -1, self.n_heads * self.d_v])
        

        output_v = self.dropout(self.proj_v(context_v), training=training)
        output_s = self.dropout(self.proj_s(context_s), training=training)
        
        return (self.layer_norm_v(residual_v + output_v),
                self.layer_norm_s(residual_s + output_s))

class co_attention(layers.Layer):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super().__init__()
        self.multi_head = MultiHeadAttention(
            d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout,
            d_model=d_model, visual_len=visual_len, sen_len=sen_len,
            fea_v=fea_v, fea_s=fea_s, pos=pos
        )
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model, 128, dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model, 128, dropout)
        
    def call(self, v, s, v_len, s_len, training=False):
        v, s = self.multi_head(v, s, v_len, s_len, training=training)
        v = self.PoswiseFeedForwardNet_v(v, training=training)
        s = self.PoswiseFeedForwardNet_s(s, training=training)
        return v, s