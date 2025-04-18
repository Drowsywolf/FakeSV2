import tensorflow as tf
from tensorflow.keras import layers


# 梯度反转层（对应ReverseLayerF）
class GradientReversal(layers.Layer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def call(self, inputs):
        return inputs
    
    def get_config(self):
        return {"alpha": self.alpha}

    def compute_output_shape(self, input_shape):
        return input_shape

    # █████ 核心：自定义梯度
    @tf.custom_gradient
    def _gradient_reversal(op):
        def grad(dy):
            return -dy * self.alpha
        return op, grad


class Attention(layers.Layer):
    def __init__(self, dim, heads=2, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # █████ PyTorch对应关系：
        # nn.Linear -> layers.Dense
        # nn.Softmax -> tf.nn.softmax
        self.to_qkv = layers.Dense(inner_dim * 3, use_bias=False)
        self.attend = tf.nn.softmax
        
        if self.project_out:
            self.to_out = tf.keras.Sequential([
                layers.Dense(dim),
                layers.Dropout(dropout)
            ])
        else:
            self.to_out = tf.identity

    def call(self, x, training=False):
        # 生成QKV [原始PyTorch代码对应]
        # torch: qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)  # █████ 分割方式变化

        # 多头重组 [Einops替代实现]
        # torch: rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
        def rearrange_head(t):
            shape = tf.shape(t)
            return tf.reshape(
                t, 
                shape=[shape[0], shape[1], self.heads, -1]
            )
            # 调整维度顺序 [b n h d] -> [b h n d]
            return tf.transpose(reshaped, [0, 2, 1, 3])
        
        q = rearrange_head(q)  # (b, h, n, d)
        k = rearrange_head(k)
        v = rearrange_head(v)

        # 注意力计算 [核心逻辑保持]
        # torch: dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = tf.matmul(q, k, transpose_b=True) * self.scale  # █████ 转置方式变化
        attn = self.attend(dots, axis=-1)  # 确保softmax在正确维度

        # 注意力加权求和
        out = tf.matmul(attn, v)
        
        # 合并多头 [Einops逆操作]
        # torch: rearrange(out, 'b h n d -> b n (h d)')
        out = tf.transpose(out, [0, 2, 1, 3])  # (b, n, h, d)
        shape = tf.shape(out)
        merged = tf.reshape(
            out, 
            [shape[0], shape[1], self.heads * dim_head]
        )

        return self.to_out(merged, training=training)  # █████ 传递training参数