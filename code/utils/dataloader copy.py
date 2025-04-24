

import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertTokenizer

import math

# 中文单位换算数字
def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif '亿' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)
        

class SVFENDDataset(Dataset):

    def __init__(self, path_vid, datamode='title+ocr'):
        
        with open('./data/dict_vid_audioconvfea.pkl', "rb") as fr:
            self.dict_vid_convfea = pickle.load(fr)

        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)
        self.data_complete = self.data_complete[self.data_complete['label']!=2] # label: 0-real, 1-fake, 2-debunk

        self.framefeapath='./dataset/ptvgg19_frames/'
        self.c3dfeapath='./dataset/c3d/'

        self.vid = []
        
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)  

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.datamode = datamode
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx): # idx: "train", "val", "test"
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label 
        label = 0 if item['annotation']=='真' else 1
        label = torch.tensor(label)

        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['description']+' '+item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['description'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # comments
        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = torch.LongTensor(np.array(comments_inputid)) 
        comments_mask = torch.LongTensor(np.array(comments_mask))
        
        comments_like = []
        for num in item['comments_like']:
            num_like = num.split(" ")[0] 
            comments_like.append(str2num(num_like))
        comments_like = torch.tensor(comments_like)
        
        # audio
        audioframes = self.dict_vid_convfea[vid]
        audioframes = torch.FloatTensor(audioframes)
        
        # frames
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)
        
        # video
        c3d = h5py.File(self.c3dfeapath+vid+".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        # # user
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
                intro = intro + '   ' + item[key]
            except:
                intro += '  '
        intro_tokens = self.tokenizer(intro, max_length=50, padding='max_length', truncation=True)
        intro_inputid = torch.LongTensor(intro_tokens['input_ids'])
        intro_mask = torch.LongTensor(intro_tokens['attention_mask'])

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'intro_inputid': intro_inputid,
            'intro_mask': intro_mask,
        }


def split_word(df):
    title = df['description'].values
    comments = df['comments'].apply(lambda x:' '.join(x)).values
    text = np.concatenate([title, comments],axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus
