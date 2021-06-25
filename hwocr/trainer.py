import os
import time
import yaml
import torch
import pickle
import numpy as np

from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch import nn
from Levenshtein import distance

from htrl.dataset import ImageDataset
from htrl.crnn import CRNN
from htrl.fcnn import FCNN
from htrl.utils import strLabelConverter

class Trainer():

    def __init__(self, device, save_folder,
                 train_dataset, valid_dataset, test1_dataset, test2_dataset, num_imgs=-1,
                 model_params={}, num_workers=8, batch_size=4, seed=34, max_epochs=1, model_pretrain='',
                 scheduler_params={}, optimizer_params={}):
        super().__init__()

        self.device = torch.device(device)
        self.save_folder = save_folder

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test1_dataset = test1_dataset
        self.test2_dataset = test2_dataset
        self.num_imgs = num_imgs
        self.num_workers = num_workers

        self.seed = seed
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        self.model_params = model_params
        
        if self.model_params['name'] == 'FCNN':
            self.model = FCNN(**self.model_params).to(self.device)
        elif self.model_params['name'] == 'CRNN':
            self.model = CRNN(**self.model_params).to(self.device)
        else:
            name = self.model_params['name']
            raise NotImplementedError(f'Model {name} not implemented.')
            
        if model_pretrain is not '':
            self.model.load_state_dict(torch.load(model_pretrain, map_location=self.device))
            print(f'Successfully loaded model from {model_pretrain}')
        self.loss = nn.CTCLoss()
        self.optimizer_params = optimizer_params
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), **self.optimizer_params)
        self.scheduler_params = scheduler_params
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.scheduler_params)

        self.initialize_training()
        self.prepare_dirs()

    def initialize_training(self):
        self.patience = 0
        self.epoch = 0
        self.score_best = None

        self.fix_seeds()
        
        train_dataset = ImageDataset(**self.train_dataset)
        self.train_iterator = DataLoader(dataset=train_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        
        valid_dataset = ImageDataset(**self.valid_dataset)
        self.valid_iterator = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        
        test1_dataset = ImageDataset(**self.test1_dataset)
        self.test1_iterator = DataLoader(dataset=test1_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        
        test2_dataset = ImageDataset(**self.test2_dataset)
        self.test2_iterator = DataLoader(dataset=test2_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)

        with open(self.train_dataset['pickle_file'], 'rb') as f:
            full_dataset = pickle.load(f)
        alphabet = ''
        for example in full_dataset:
            alphabet += example['description']
        alphabet = list(set(alphabet))
        self.alphabet =''.join(alphabet)
        self.converter = strLabelConverter(self.alphabet)


    def prepare_dirs(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def get_parameters(self):
        params = {
                    "device": str(self.device),
                    "save_folder": self.save_folder,
                    "train_dataset": self.train_dataset,
                    "valid_dataset": self.valid_dataset,
                    "test1_dataset": self.test1_dataset,
                    "test2_dataset": self.test2_dataset,
                    "num_imgs": self.num_imgs,
                    "num_workers": self.num_workers,
                    "seed": self.seed,
                    "batch_size": self.batch_size,
                    "model_params": self.model_params,
                    "max_epochs": self.max_epochs,
        }
        return params

    def fix_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
    def train_on_batches(self):
        self.model.train()
        while True:
            with tqdm(total=len(self.train_iterator)) as bar_train:
                for x, y_true in self.train_iterator:
                    '''TRAINING CODE HERE'''
                    pred_text = self.model(x.to(self.device))
                    preds_size = torch.LongTensor([pred_text.size(0)] * self.batch_size)

                    t_text, l_text = self.converter.encode(list(y_true))

                    loss = self.loss(pred_text, t_text, preds_size, l_text)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    _, decode_text = pred_text.max(2)
                    decode_text = decode_text.transpose(1, 0).contiguous().view(-1)
                    decode_text = self.converter.decode(decode_text.data, preds_size, raw=False)

                    char_correct = 0
                    total_char = 0
                    for pred, gt in zip(decode_text, y_true):
                        for i, pred_char in enumerate(pred):
                            if i<len(gt):
                                if pred_char == gt[i]:
                                    char_correct += 1
                        total_char += len(gt)

                    acc_char = (char_correct) / float(total_char)

                    total_str = 0
                    n_correct = 0
                    cer = 0
                    for pred, target in zip(decode_text, y_true):
                        if pred == target:
                            n_correct += 1
                        else:
                            cer += distance(pred, target)/len(target)
                        total_str+=1
                    acc_str = (n_correct) / total_str
                    cer = cer/total_str
                    '''TRAINING CODE HERE'''
                    printed_data = f"Ep: {self.epoch}. Train loss: {float(loss.item()):.4f}.\
                                     CER: {cer:.4f}."
                    bar_train.set_description(printed_data)
                    bar_train.update(1)

            self.epoch += 1
            self.scheduler.step()
            self.validate()
            if 0 < self.max_epochs <= self.epoch:
                break
        
        
    def train(self):
        try:
            self.train_on_batches()
        except KeyboardInterrupt:
            print("Stopped training")
        finally:
            self.save(save_policy='last')
    
    def validate(self):
        val_epoch_loss = []
        val_epoch_acc_char = []
        val_epoch_acc_str = []
        val_epoch_cer = []
        with tqdm(total = len(self.valid_iterator)) as bar_val:
            for i, (images, gt_text) in enumerate(self.valid_iterator):
                results = self.validate_on_batches(images, gt_text)
                loss, acc_char, acc_str, cer, decode_text, gt_text, raw_preds = results
                
                val_epoch_loss.append(loss)
                val_epoch_acc_char.append(acc_char)
                val_epoch_acc_str.append(acc_str)
                val_epoch_cer.append(cer)
                
                printed_data = f"Ep: {self.epoch}. Val loss: {np.mean(val_epoch_loss):.4f}.\
                Acc char: {np.mean(val_epoch_acc_char):.4f}.\
                Acc str: {np.mean(val_epoch_acc_str):.4f}.\
                CER: {np.mean(val_epoch_cer):.4f}."
                bar_val.set_description(printed_data)
                bar_val.update(1)
        if self.score_best is None:
            self.score_best = np.mean(val_epoch_loss)
        else:
            if self.score_best > np.mean(val_epoch_loss):
                self.score_best = np.mean(val_epoch_loss)
                self.save(save_policy='best')
                print(f'Model improved on valid with loss: {np.mean(val_epoch_loss)}')
        for raw_pred, pred, gt in zip(raw_preds, decode_text, gt_text):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt)) 
        self.model.train()
            
    def validate_on_batches(self, images, gt_text):
        
        self.model.eval()
        pred_text = self.model(images.to(self.device))

        preds_size = torch.LongTensor([pred_text.size(0)] * self.batch_size)

        t_text, l_text = self.converter.encode(list(gt_text))

        loss = self.loss(pred_text, t_text, preds_size, l_text)

        _, decode_text = pred_text.max(2)
        decode_text = decode_text.transpose(1, 0).contiguous().view(-1)
        decode_text = self.converter.decode(decode_text.data, preds_size, raw=False)

        char_correct = 0
        total_char = 0
        for pred, gt in zip(decode_text, gt_text):
            for i, pred_char in enumerate(pred):
                if i<len(gt):
                    if pred_char == gt[i]:
                        char_correct += 1
            total_char += len(gt)

        acc_char = (char_correct)/float(total_char)

        total_str = 0
        n_correct = 0
        cer = 0
        for pred, target in zip(decode_text, gt_text):
            if pred == target:
                n_correct += 1
            else:
                cer += distance(pred, target)/len(target)
            total_str+=1

        acc_str = (n_correct)/total_str
        cer = cer/total_str

        _, preds = pred_text.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        raw_preds = self.converter.decode(preds.data, preds_size, raw=True)

        return (loss.item(), acc_char, acc_str, cer, decode_text[:5], gt_text[:5], raw_preds[:5])

    def save(self, save_policy='best'):
        print(f"Saving trainer to {self.save_folder}.")
        if len(self.save_folder) > 0 and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            
        if save_policy == 'best':
            torch.save(self.model.state_dict(), os.path.join(self.save_folder, "model_state_dict"))
            torch.save(self.model, os.path.join(self.save_folder, "model"))
        elif save_policy == 'last':
            torch.save(self.model.state_dict(), os.path.join(self.save_folder, "model_last_state_dict"))
            torch.save(self.model, os.path.join(self.save_folder, "model_last"))

        torch.save({
            "parameters": self.get_parameters()
        }, os.path.join(self.save_folder, "trainer"))
        print("Trainer is saved.")

    @classmethod
    def load(cls, load_folder, device="cpu", load_policy='best'):
        checkpoint = torch.load(os.path.join(load_folder, "trainer"), map_location=device)
        parameters = checkpoint["parameters"]
        parameters.pop("device", None)
        trainer = cls(device=device, **parameters)
        if load_policy == 'best':
            trainer.model = torch.load(os.path.join(load_folder, "model"))
        elif load_policy == 'last':
            trainer.model = torch.load(os.path.join(load_folder, "model_last"))
        return trainer
