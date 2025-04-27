import copy
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from transformers import TFBertModel, BertModel, AutoModelForMaskedLM

from .coattention import *
from .layers import *
from utils.metrics import *

class SVFENDModel(tf.keras.Model):
    def __init__(self, bert_model, fea_dim, dropout):
        super(SVFENDModel, self).__init__()

        print("SVFENDModel, init")

        self.bert = TFBertModel.from_pretrained(bert_model)

        self.text_dim = 48 #768
        self.comment_dim = 48 #768
        self.img_dim = 256 #4096
        self.video_dim = 256 #4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4

        self.dropout_rate = dropout

        self.attention = Attention(dim=self.dim, heads=4, dropout=dropout)

        # You need a TensorFlow compatible VGGish model here
        self.vggish_modified = tf.keras.Sequential([
            tf.keras.layers.Dense(128)  # Mock layer; replace with your VGGish structure
        ])

        self.co_attention_ta = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout_rate, d_model=fea_dim,
                                           visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_tv = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout_rate, d_model=fea_dim,
                                           visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)

        self.linear_text = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])
        self.linear_comment = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])
        self.linear_img = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])
        self.linear_video = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])
        self.linear_intro = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])
        self.linear_audio = tf.keras.Sequential([
            tf.keras.layers.Dense(fea_dim), tf.keras.layers.ReLU(), tf.keras.layers.Dropout(dropout)
        ])

        self.classifier = tf.keras.layers.Dense(2)

        # Transformer encoder (simple version)
        self.encoder_layer = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=self.dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dim*4, activation='relu'),
            tf.keras.layers.Dense(self.dim)
        ])

    def call(self, inputs):

        print("SVFENDModel, call")

        intro_inputid = inputs['intro_inputid']
        intro_mask = inputs['intro_mask']
        fea_intro = self.bert(intro_inputid, attention_mask=intro_mask).pooler_output
        fea_intro = self.linear_intro(fea_intro)

        title_inputid = inputs['title_inputid']
        title_mask = inputs['title_mask']
        fea_text = self.bert(title_inputid, attention_mask=title_mask).last_hidden_state
        fea_text = self.linear_text(fea_text)

        audioframes = inputs['audioframes']
        fea_audio = self.vggish_modified(audioframes)
        fea_audio = self.linear_audio(fea_audio)
        fea_audio, fea_text = self.co_attention_ta(fea_audio, fea_text, tf.shape(fea_audio)[1], tf.shape(fea_text)[1])
        fea_audio = tf.reduce_mean(fea_audio, axis=1)

        frames = inputs['frames']
        fea_img = self.linear_img(frames)
        fea_img, fea_text = self.co_attention_tv(fea_img, fea_text, tf.shape(fea_img)[1], tf.shape(fea_text)[1])
        fea_img = tf.reduce_mean(fea_img, axis=1)

        fea_text = tf.reduce_mean(fea_text, axis=1)

        c3d = inputs['c3d']
        fea_video = self.linear_video(c3d)
        fea_video = tf.reduce_mean(fea_video, axis=1)

        comments_inputid = inputs['comments_inputid']
        comments_mask = inputs['comments_mask']
        comments_like = inputs['comments_like']

        comments_feature = []
        for i in range(tf.shape(comments_inputid)[0]):
            bert_fea = self.bert(comments_inputid[i], attention_mask=comments_mask[i]).pooler_output
            comments_feature.append(bert_fea)
        comments_feature = tf.stack(comments_feature)

        fea_comments = []
        for v in range(tf.shape(comments_like)[0]):
            weights = tf.cast(tf.range(1, tf.shape(comments_like[v])[0]+1), dtype=tf.float32)
            comments_weight = weights / (tf.cast(tf.shape(comments_like[v])[0], dtype=tf.float32) + tf.reduce_sum(tf.cast(comments_like[v], dtype=tf.float32)))
            comments_fea_reweight = tf.reduce_sum(comments_feature[v] * tf.expand_dims(comments_weight, -1), axis=0)
            fea_comments.append(comments_fea_reweight)
        fea_comments = tf.stack(fea_comments)
        fea_comments = self.linear_comment(fea_comments)

        fea_text = tf.expand_dims(fea_text, 1)
        fea_comments = tf.expand_dims(fea_comments, 1)
        fea_img = tf.expand_dims(fea_img, 1)
        fea_audio = tf.expand_dims(fea_audio, 1)
        fea_video = tf.expand_dims(fea_video, 1)
        fea_intro = tf.expand_dims(fea_intro, 1)

        fea = tf.concat([fea_text, fea_audio, fea_video, fea_intro, fea_img, fea_comments], axis=1)

        attn_output = self.mha(fea, fea)
        ffn_output = self.ffn(attn_output)
        fea = self.encoder_layer(fea + ffn_output)

        fea = tf.reduce_mean(fea, axis=1)

        output = self.classifier(fea)

        return output, fea
