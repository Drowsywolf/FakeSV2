import copy
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from tqdm import tqdm
from transformers import TFBertModel

from .coattention import *
from .layers import *
from utils.metrics import *

class Trainer3():
    def __init__(self,
                model,
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
                save_threshold=0.0,
                start_epoch=0):

        self.model = model
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

        since = time.time()

        best_model_wts_val = self.model.get_weights()
        best_acc_val = 0.0
        best_epoch_val = 0

        is_earlystop = False

        if self.mode == "eann":
            best_acc_val_event = 0.0
            best_epoch_val_event = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print(f'Epoch {epoch+1}/{self.start_epoch + self.num_epochs}')
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            tf.keras.backend.set_value(self.optimizer.lr, lr)

            for phase in ['train', 'val', 'test']:
                training = (phase == 'train')
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss = 0.0
                tpred = []
                tlabel = []

                if self.mode == "eann":
                    running_loss_event = 0.0
                    running_loss_fnd = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch_data in tqdm(self.dataloaders[phase]):
                    label = batch_data['label']
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                    with tf.GradientTape() as tape:
                        if self.mode == "eann":
                            outputs, outputs_event, fea = self.model(batch_data, training=training)
                            loss_fnd = self.criterion(label, outputs)
                            loss_event = self.criterion(label_event, outputs_event)
                            loss = loss_fnd + loss_event
                            preds = tf.argmax(outputs, axis=-1)
                            preds_event = tf.argmax(outputs_event, axis=-1)
                        else:
                            outputs, fea = self.model(batch_data, training=training)
                            preds = tf.argmax(outputs, axis=-1)
                            loss = self.criterion(label, outputs)

                    if training:
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
                print(f'Loss: {epoch_loss:.4f}')
                results = metrics(tlabel, tpred)
                print(results)
                self.writer.add_scalar('Loss/' + phase, epoch_loss, epoch + 1)
                self.writer.add_scalar('Acc/' + phase, results['acc'], epoch + 1)
                self.writer.add_scalar('F1/' + phase, results['f1'], epoch + 1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print(f'Loss_fnd: {epoch_loss_fnd:.4f}')
                    print(f'Loss_event: {epoch_loss_event:.4f}')
                    self.writer.add_scalar('Loss_fnd/' + phase, epoch_loss_fnd, epoch + 1)
                    self.writer.add_scalar('Loss_event/' + phase, epoch_loss_event, epoch + 1)

                if phase == 'val' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_model_wts_val = self.model.get_weights()
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        self.model.save_weights(self.save_param_path + f"_val_epoch{best_epoch_val}_{best_acc_val:.4f}")
                        print(f"saved {self.save_param_path}_val_epoch{best_epoch_val}_{best_acc_val:.4f}")
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Best model on val: epoch{best_epoch_val}_{best_acc_val}")

        self.model.set_weights(best_model_wts_val)
        print("test result when using best model on val")
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

        print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print(metrics(label, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print("event:")
            print(accuracy_score(np.array(label_event), np.array(pred_event)))

        return metrics(label, pred)
