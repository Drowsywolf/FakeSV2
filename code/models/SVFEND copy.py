import copy
import json
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import *
from tqdm import tqdm
from transformers import TFAutoModel, AutoConfig

from .coattention import *
from .layers import *
from utils.metrics import *

class SVFENDModel(Model):
    def __init__(self, bert_model, fea_dim, dropout):
        super(SVFENDModel, self).__init__()
        

        self.bert = TFAutoModel.from_pretrained(bert_model, trainable=False)  
        

        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4
        self.dropout = dropout


        self.attention = Attention(dim=self.dim, heads=4, dropout=dropout)
        

        self.vggish_layer = self.load_vggish_tf()  
        self.vggish_modified = tf.keras.Sequential(self.vggish_layer.layers[-2:-1])
        

        self.co_attention_ta = co_attention(
            d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=fea_dim, visual_len=self.num_audioframes, sen_len=512,
            fea_v=self.dim, fea_s=self.dim, pos=False)
        
        self.co_attention_tv = co_attention(
            d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=fea_dim, visual_len=self.num_frames, sen_len=512,
            fea_v=self.dim, fea_s=self.dim, pos=False)


        self.trm = layers.TransformerEncoder(
            num_layers=1,  
            d_model=self.dim,
            num_heads=2,
            dropout=dropout,
            activation='relu',
            norm_first=False
        )


        self.linear_text = tf.keras.Sequential([
            layers.Dense(fea_dim, activation='relu'),
            layers.Dropout(dropout)
        ])

        self.linear_comment = tf.keras.Sequential([...]) 
        self.linear_img = tf.keras.Sequential([...])
        self.linear_video = tf.keras.Sequential([...])
        self.linear_intro = tf.keras.Sequential([...])
        self.linear_audio = tf.keras.Sequential([...])


        self.classifier = layers.Dense(2)

    def call(self, inputs, training=False):
        
        fea_intro = self.bert(
            inputs['intro_inputid'], 
            attention_mask=inputs['intro_mask']
        ).pooler_output
        fea_intro = self.linear_intro(fea_intro, training=training)

        
        text_output = self.bert(
            inputs['title_inputid'],
            attention_mask=inputs['title_mask']
        )
        fea_text = self.linear_text(text_output.last_hidden_state, training=training)


        fea_audio = self.vggish_modified(inputs['audioframes'])
        fea_audio = self.linear_audio(fea_audio, training=training)

        fea_audio, fea_text = self.co_attention_ta(
            v=fea_audio, s=fea_text, 
            v_len=tf.shape(fea_audio)[1], 
            s_len=tf.shape(fea_text)[1]
        )
        fea_audio = tf.reduce_mean(fea_audio, axis=1)


        fea_img = self.linear_img(inputs['frames'], training=training)
        fea_img, fea_text = self.co_attention_tv(
            v=fea_img, s=fea_text,
            v_len=tf.shape(fea_img)[1],
            s_len=tf.shape(fea_text)[1]
        )
        fea_img = tf.reduce_mean(fea_img, axis=1)

        fea_text = tf.reduce_mean(fea_text, axis=1)


        fea_video = self.linear_video(inputs['c3d'], training=training)
        fea_video = tf.reduce_mean(fea_video, axis=1)


        comments_feature = []
        for i in range(tf.shape(inputs['comments_inputid'])[0]):
            output = self.bert(
                inputs['comments_inputid'][i],
                attention_mask=inputs['comments_mask'][i]
            )
            comments_feature.append(output.pooler_output)
        comments_feature = tf.stack(comments_feature)


        fea_comments = []
        for v in range(tf.shape(inputs['comments_like'])[0]):
            weights = tf.stack([(i+1)/(tf.shape(inputs['comments_like'][v])[0] + 
                                  tf.reduce_sum(inputs['comments_like'][v])) 
                                 for i in inputs['comments_like'][v]])
            weighted_fea = tf.reduce_sum(
                comments_feature[v] * tf.reshape(weights, (-1, 1)),
                axis=0
            )
            fea_comments.append(weighted_fea)
        fea_comments = tf.stack(fea_comments)
        fea_comments = self.linear_comment(fea_comments, training=training)


        fea_text = tf.expand_dims(fea_text, 1)
        fea_comments = tf.expand_dims(fea_comments, 1)
        fea_img = tf.expand_dims(fea_img, 1)
        fea_audio = tf.expand_dims(fea_audio, 1)
        fea_video = tf.expand_dims(fea_video, 1)
        fea_intro = tf.expand_dims(fea_intro, 1)

        fea = tf.concat([fea_text, fea_audio, fea_video, fea_intro,
                        fea_img, fea_comments], axis=1)
        fea = self.trm(fea, training=training)
        fea = tf.reduce_mean(fea, axis=1)
        
        output = self.classifier(fea)
        return output, fea