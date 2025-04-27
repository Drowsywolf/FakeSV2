import os
import pickle
import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
import torch
import torch.nn as nn

def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1]) * 10000
    elif '亿' in str_x:
        return float(str_x[:-1]) * 100000000
    else:
        print("error")
        print(str_x)

class SVFENDDataset():
    def __init__(self, path_vid, datamode='title+ocr'):
        print("SVFENDDataset, init")
        with open('./dataset/dict_vid_audioconvfea.pkl', "rb") as fr:
            self.dict_vid_convfea = pickle.load(fr)

        print("Loading data...")
        self.data_complete = pd.read_json('./dataset/data_complete_100.json', orient='records', dtype=False, lines=True)
        self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']

        self.framefeapath = './dataset/ptvgg19_frames/ptvgg19_frames/'
        self.c3dfeapath = './dataset/c3d/c3d/'

        print("Loading video list...")
        self.vid = []
        with open('./dataset/data-split/event/' + path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        
        print("Loading video features...")
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)].copy()
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        print(1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.datamode = datamode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label
        label = 0 if item['annotation'] == '真' else 1
        label = tf.convert_to_tensor(label, dtype=tf.int32)

        # text
        if self.datamode == 'title+ocr':
            text = item['title'] + ' ' + item['ocr']
        elif self.datamode == 'ocr':
            text = item['ocr']
        elif self.datamode == 'title':
            text = item['title']
        title_tokens = self.tokenizer(text, max_length=512, padding='max_length', truncation=True)
        title_inputid = tf.convert_to_tensor(title_tokens['input_ids'], dtype=tf.int32)
        title_mask = tf.convert_to_tensor(title_tokens['attention_mask'], dtype=tf.int32)

        # comments
        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = tf.convert_to_tensor(np.array(comments_inputid), dtype=tf.int32)
        comments_mask = tf.convert_to_tensor(np.array(comments_mask), dtype=tf.int32)
        # comments_inputid = torch.LongTensor(np.array(comments_inputid)) 
        # comments_mask = torch.LongTensor(np.array(comments_mask))


        comments_like = []
        for num in item['count_comment_like']:
            num_like = num.split(" ")[0]
            comments_like.append(str2num(num_like))
        comments_like = tf.convert_to_tensor(comments_like, dtype=tf.float32)
        # comments_like = torch.tensor(comments_like)

        # audio frames
        audioframes = self.dict_vid_convfea[vid]
        audioframes = tf.convert_to_tensor(audioframes, dtype=tf.float32)

        # frames
        frames = pickle.load(open(os.path.join(self.framefeapath, vid + '.pkl'), 'rb'))
        frames = tf.convert_to_tensor(frames, dtype=tf.float32)

        # c3d video features
        c3d = h5py.File(self.c3dfeapath + vid + ".hdf5", "r")[vid]['c3d_features'][:]
        c3d = tf.convert_to_tensor(c3d, dtype=tf.float32)

        # user intro
        try:
            if item['is_official'] == 1:
                intro = "个人认证"
            elif item['is_official'] == 2:
                intro = "机构认证"
            elif item['is_official'] == 0:
                intro = "未认证"
            else:
                intro = "认证状态未知"
        except:
            intro = "认证状态未知"

        for key in ['poster_intro', 'content_verify']:
            try:
                intro += '   ' + item[key]
            except:
                intro += '  '
        intro_tokens = self.tokenizer(intro, max_length=50, padding='max_length', truncation=True)
        intro_inputid = tf.convert_to_tensor(intro_tokens['input_ids'], dtype=tf.int32)
        intro_mask = tf.convert_to_tensor(intro_tokens['attention_mask'], dtype=tf.int32)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'frames': frames,
            'c3d': c3d,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'intro_inputid': intro_inputid,
            'intro_mask': intro_mask,
        }

    def get_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            lambda: (self[i] for i in range(len(self))),
            output_signature={
                'label': tf.TensorSpec(shape=(), dtype=tf.int32),
                'title_inputid': tf.TensorSpec(shape=(512,), dtype=tf.int32),
                'title_mask': tf.TensorSpec(shape=(512,), dtype=tf.int32),
                'audioframes': tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
                'frames': tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
                'c3d': tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
                'comments_inputid': tf.TensorSpec(shape=(None, 250), dtype=tf.int32),
                'comments_mask': tf.TensorSpec(shape=(None, 250), dtype=tf.int32),
                'comments_like': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'intro_inputid': tf.TensorSpec(shape=(50,), dtype=tf.int32),
                'intro_mask': tf.TensorSpec(shape=(50,), dtype=tf.int32),
            }
        )

def split_word(df):
    title = df['description'].values
    comments = df['comments'].apply(lambda x: ' '.join(x)).values
    text = np.concatenate([title, comments], axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus
