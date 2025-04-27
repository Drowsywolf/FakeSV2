import copy
import json
import os
import time
from tkinter import E

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import BertModel
from utils.metrics import *
from zmq import device

from .coattention import *
from .layers import *



class Trainer():
    def __init__(self,
                model, 
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 mode,
                 model_name, 
                 event_num,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        print("Trainer, init")

        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if not os.path.exists(save_param_path):
            os.makedirs(save_param_path)
        self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        

    def train(self):
        print("Training...")

        since = time.time()


        # best_model_wts_test = self.model.get_weights()
        # best_model_wts_test = copy.deepcopy(self.model.get_weights())
        best_acc_test = 0.0
        best_epoch_test = 0

        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            tf.keras.backend.set_value(self.optimizer.lr, lr)
            
            for phase in ['train', 'test']:
                training = (phase == 'train')  
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0 
                tpred = []
                tlabel = []

                if self.mode == "eann":
                    running_loss_event = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    for k,v in batch_data.items():
                        batch_data[k]=v
                    label = batch_data['label']
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                
                    with tf.GradientTape() as tape:
                        if self.mode == "eann":
                            outputs, outputs_event,fea = self.model(batch_data, training=training)
                            loss_fnd = self.criterion(outputs, label)
                            loss_event = self.criterion(outputs_event, label_event)
                            loss = loss_fnd + loss_event
                            preds = tf.argmax(outputs, axis=-1)
                            preds_event = tf.argmax(outputs_event, axis=-1)
                        else:
                            outputs,fea = self.model(batch_data, training=training)
                            preds = tf.argmax(outputs, axis=-1)
                            loss = self.criterion(outputs, label)

                        if phase == 'train':
                            gradients = tape.gradient(loss, self.model.trainable_variables)
                            gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
                            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    tlabel.extend(label.numpy().tolist())
                    tpred.extend(preds.numpy().tolist())
                    running_loss += loss.numpy() * label.shape[0]

                    if self.mode == "eann":
                        tlabel_event.extend(label_event.numpy().tolist())
                        tpred_event.extend(preds_event.numpy().tolist())
                        running_loss_event += loss_event.numpy() * label_event.shape[0]
                        running_loss_fnd += loss_fnd.numpy() * label.shape[0]
                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/'+phase, epoch_loss_fnd, epoch+1)
                    self.writer.add_scalar('Loss_event/'+phase, epoch_loss_event, epoch+1)
                
                if phase == 'test':
                    if results['acc'] > best_acc_test:
                        best_acc_test = results['acc']
                        best_model_wts_test = self.model.get_weights()
                        best_epoch_test = epoch + 1
                        if best_acc_test > self.save_threshold:
                            self.model.save_weights(self.save_param_path + f"_val_epoch{best_epoch_test}_{best_acc_test:.4f}")
                            print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test) )
                    else:
                        if epoch-best_epoch_test >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on test: epoch" + str(best_epoch_test) + "_" + str(best_acc_test))

        if self.mode == "eann":
            print("Event: Best model on test: epoch" + str(best_epoch_test_event) + "_" + str(best_acc_test_event))

        self.model.set_weights(best_model_wts_test)
        return self.test()


    def test(self):
        since = time.time()

        pred = []
        label = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch_data in tqdm(self.dataloaders['test']):
                label_batch = batch_data['label']

                if self.mode == "eann":
                    label_event_batch = batch_data['label_event']
                    outputs, outputs_event, fea = self.model(batch_data, training=False)
                    preds_event = tf.argmax(outputs_event, axis=-1)
                    label_event.extend(label_event_batch.numpy().tolist())
                    pred_event.extend(preds_event.numpy().tolist())
                else: 
                    outputs, fea = self.model(batch_data, training=False)

                preds = tf.argmax(outputs, axis=-1)
                label.extend(label_batch.numpy().tolist())
                pred.extend(preds.numpy().tolist())

        print (get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print (metrics(label, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print ("event:")
            print (accuracy_score(np.array(label_event), np.array(pred_event)))

        return metrics(label, pred)

