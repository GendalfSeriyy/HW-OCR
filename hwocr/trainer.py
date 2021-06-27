import os
import time
import yaml
import torch
import pickle
import random
import numpy as np
import itertools

from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from Levenshtein import distance

from hwocr.dataset import ImageDataset
from hwocr.crnn import CRNN
from hwocr.fcnn import FCNN
from hwocr.attention_crnn import Encoder, Decoder
from hwocr.utils import strLabelConverter, ConvertBetweenStringAndLabel, SOS_TOKEN, EOS_TOKEN
from hwocr.metrics import get_word_error_rate

class Trainer():

    def __init__(self, device, save_folder,
                 train_dataset, valid_dataset, test1_dataset, test2_dataset, num_imgs=-1,
                 model_params={}, num_workers=8, batch_size=4, seed=34, max_epochs=1, model_pretrain='', teach_forcing_prob=0.5,
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
        self.teach_forcing_prob = teach_forcing_prob

        self.seed = seed
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        self.model_params = model_params
        
        if self.model_params['name'] == 'FCNN':
            self.model = FCNN(**self.model_params).to(self.device)
            self.encoder, self.decoder = None, None
        elif self.model_params['name'] == 'CRNN':
            self.model = CRNN(**self.model_params).to(self.device)
            self.encoder, self.decoder = None, None
        elif self.model_params['name'] == 'AttenCRNN':
            self.encoder = Encoder(**self.model_params['Encoder']).to(self.device)
            self.decoder = Decoder(**self.model_params['Decoder']).to(self.device)
            self.model = None
        else:
            name = self.model_params['name']
            raise NotImplementedError(f'Model {name} not implemented.')
            
        if model_pretrain is not '':
            self.model.load_state_dict(torch.load(model_pretrain, map_location=self.device))
            print(f'Successfully loaded model from {model_pretrain}')
            
        if self.model_params['name'] == 'CRNN' or self.model_params['name'] == 'FCNN':
            self.loss = nn.CTCLoss()
        else:
            self.loss = torch.nn.NLLLoss()
        self.optimizer_params = optimizer_params
        if self.encoder is not None and self.decoder is not None:
            self.optimizer = getattr(optim, self.optimizer_params["class"])(
                itertools.chain(self.encoder.parameters(),
                                self.decoder.parameters()),
                **self.optimizer_params["parameters"]
            )
        else:
            self.optimizer = getattr(optim, self.optimizer_params["class"])(
                self.model.parameters(),
                **self.optimizer_params["parameters"]
            )

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
        if self.model_params['name'] == 'AttenCRNN':
            self.valid_iterator = DataLoader(dataset=valid_dataset, batch_size=1, 
                                             shuffle=True, num_workers=self.num_workers, drop_last = True)
        else:
            self.valid_iterator = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        
        test1_dataset = ImageDataset(**self.test1_dataset)
        if self.model_params['name'] == 'AttenCRNN':
            self.test1_iterator = DataLoader(dataset=test1_dataset, batch_size=1, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        else:
            self.test1_iterator = DataLoader(dataset=test1_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        
        test2_dataset = ImageDataset(**self.test2_dataset)

        if self.model_params['name'] == 'AttenCRNN':
            self.test2_iterator = DataLoader(dataset=test2_dataset, batch_size=1, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)
        else:
            self.test2_iterator = DataLoader(dataset=test2_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=self.num_workers, drop_last = True)

        with open(self.train_dataset['pickle_file'], 'rb') as f:
            full_dataset = pickle.load(f)
        alphabet = ''
        for example in full_dataset:
            alphabet += example['description']
        alphabet = list(set(alphabet))
        self.alphabet =''.join(alphabet)
        if self.model_params['name'] == 'CRNN' or self.model_params['name'] == 'FCNN':
            self.converter = strLabelConverter(self.alphabet)
        else:
            self.converter = ConvertBetweenStringAndLabel(self.alphabet)


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
                    "teach_forcing_prob": self.teach_forcing_prob,
                    "scheduler_params": self.scheduler_params,
                    "optimizer_params": self.optimizer_params,
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
            train_loss_epoch = []
            CER_epoch = []
            WER_epoch = []
            with tqdm(total=len(self.train_iterator), ncols=900) as bar_train:
                for x, y_true in self.train_iterator:

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

                    total_str = 0
                    n_correct = 0
                    cer = 0
                    wer = 0
                    for pred, target in zip(decode_text, y_true):
                        if pred == target:
                            n_correct += 1
                        else:
                            cer += distance(pred, target)/len(target)
                            wer += get_word_error_rate(target.split(' '), pred.split(' '))
                        total_str+=1
                    acc_str = (n_correct) / total_str
                    cer = cer/total_str
                    wer = wer/total_str
                    
                    train_loss_epoch.append(loss.item())
                    CER_epoch.append(cer)
                    WER_epoch.append(wer)

                    printed_data = f"Ep: {self.epoch}. Train loss: {np.mean(train_loss_epoch):.4f}.\
                                     CER: {np.mean(CER_epoch):.4f}.\
                                     WER: {np.mean(wer):.4f}."
                    bar_train.set_description(printed_data)
                    bar_train.update(1)

            self.epoch += 1
            self.scheduler.step()
            self.validate()
            if 0 < self.max_epochs <= self.epoch:
                break
                
    def train_on_batches_attention(self):
        self.encoder.train()
        self.decoder.train()
        while True:
            train_loss_epoch = []
            with tqdm(total=len(self.train_iterator), ncols=900) as bar_train:
                for x, y_true in self.train_iterator:

                    batch_size = x.size(0)
                    self.encoder.train()
                    self.decoder.train()

                    target_variable = self.converter.encode(y_true)

                    # CNN + BiLSTM
                    encoder_outputs = self.encoder(x.to(self.device))
                    target_variable = target_variable.to(self.device)
                    # start decoder for SOS_TOKEN
                    decoder_input = target_variable[SOS_TOKEN].to(self.device)
                    decoder_hidden = self.decoder.initHidden(batch_size).to(self.device)

                    decoded_labels = []

                    loss = 0.0
                    if random.random() > self.teach_forcing_prob:
                        teach_forcing = True
                    else:
                        teach_forcing = False
                    if teach_forcing:
                        for di in range(1, target_variable.shape[0]):
                            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                            loss += self.loss(decoder_output, target_variable[di])
                            decoder_input = target_variable[di]
                    else:
                        for di in range(1, target_variable.shape[0]):
                            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                            loss += self.loss(decoder_output, target_variable[di])
                            topv, topi = decoder_output.data.topk(1)
                            ni = topi.squeeze()
                            decoder_input = ni

                    self.encoder.zero_grad()
                    self.decoder.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss_epoch.append(loss.item())
                    
                    printed_data = f"Ep: {self.epoch}. Train loss: {np.mean(train_loss_epoch):.4f}."
                    bar_train.set_description(printed_data)
                    bar_train.update(1)

            self.epoch += 1
            self.scheduler.step()
            self.validate()
            if 0 < self.max_epochs <= self.epoch:
                break
        
        
    def train(self):
        try:
            if self.model_params['name'] == 'AttenCRNN':
                self.train_on_batches_attention()
            else:
                self.train_on_batches()
        except KeyboardInterrupt:
            print("Stopped training")
        finally:
            self.save(save_policy='last')
    
    def validate(self, iterator = None, test = False):
        val_epoch_loss = []
        val_epoch_acc_str = []
        val_epoch_cer = []
        val_epoch_wer = []
        if iterator==None:
            iterator = self.valid_iterator
        with tqdm(total = len(iterator), ncols=900) as bar_val:
            for i, (images, gt_text) in enumerate(iterator):
                if self.model_params['name'] == 'AttenCRNN':
                    results = self.validate_on_batches_attention(images, gt_text)
                    loss, accuracy, cer, wer = results
                    val_epoch_loss.append(loss)
                    val_epoch_cer.append(cer)
                    val_epoch_wer.append(wer)
                    
                    printed_data = f"Ep: {self.epoch-1}. Val loss: {np.mean(val_epoch_loss):.4f}.\
                    CER: {np.mean(val_epoch_cer):.4f}.\
                    WER: {np.mean(val_epoch_wer):.4f}."
                    bar_val.set_description(printed_data)
                    bar_val.update(1)
                else:
                    results = self.validate_on_batches(images, gt_text)
                    loss, acc_str, cer, wer, decode_text, gt_text, raw_preds = results

                    val_epoch_loss.append(loss)
                    val_epoch_acc_str.append(acc_str)
                    val_epoch_cer.append(cer)
                    val_epoch_wer.append(wer)

                    printed_data = f"Ep: {self.epoch-1}. Val loss: {np.mean(val_epoch_loss):.4f}.\
                    Acc str: {np.mean(val_epoch_acc_str):.4f}.\
                    CER: {np.mean(val_epoch_cer):.4f}.\
                    WER: {np.mean(val_epoch_wer):.4f}."
                    bar_val.set_description(printed_data)
                    bar_val.update(1)
        if not test:
            if self.score_best is None:
                self.score_best = np.mean(val_epoch_loss)
            else:
                if self.score_best > np.mean(val_epoch_loss):
                    self.score_best = np.mean(val_epoch_loss)
                    self.save(save_policy='best')
                    print(f'Model improved on valid with loss: {np.mean(val_epoch_loss)}')
        if self.model_params['name'] != 'AttenCRNN':
            for raw_pred, pred, gt in zip(raw_preds, decode_text, gt_text):
                print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt)) 
        print('Lr now:', self.optimizer.param_groups[0]['lr'])
        if test:
            print('CER:', np.mean(val_epoch_cer))
            print('WER:', np.mean(val_epoch_wer))
        if self.model_params['name'] != 'AttenCRNN':
            self.model.train()
        else:
            self.encoder.train()
            self.decoder.train()
            
    def validate_on_batches(self, images, gt_text):
        
        self.model.eval()
        pred_text = self.model(images.to(self.device))

        preds_size = torch.LongTensor([pred_text.size(0)] * self.batch_size)

        t_text, l_text = self.converter.encode(list(gt_text))

        loss = self.loss(pred_text, t_text, preds_size, l_text)

        _, decode_text = pred_text.max(2)
        decode_text = decode_text.transpose(1, 0).contiguous().view(-1)
        decode_text = self.converter.decode(decode_text.data, preds_size, raw=False)

        total_str = 0
        n_correct = 0
        cer = 0
        wer = 0
        for pred, target in zip(decode_text, gt_text):
            if pred == target:
                n_correct += 1
            else:
                cer += distance(pred, target)/len(target)
                wer += get_word_error_rate(target.split(' '), pred.split(' '))
            total_str+=1
        acc_str = (n_correct) / total_str
        cer = cer/total_str
        wer = wer/total_str

        _, preds = pred_text.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        raw_preds = self.converter.decode(preds.data, preds_size, raw=True)

        return (loss.item(), acc_str, cer, wer, decode_text[:5], gt_text[:5], raw_preds[:5])
    
    def validate_on_batches_attention(self, images, gt_text):
        
        self.encoder.eval()
        self.decoder.eval()

        n_correct = 0
        n_total = 0

        batch_size = images.size(0)

        target_variable = self.converter.encode(gt_text)
        n_total += len(gt_text[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = self.encoder(images.to(self.device))
        target_variable = target_variable.to(self.device)
        decoder_input = target_variable[0].to(self.device)
        decoder_hidden = self.decoder.initHidden(batch_size).to(self.device)

        loss = 0.
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += self.loss(decoder_output, target_variable[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == EOS_TOKEN:
                decoded_label.append(EOS_TOKEN)
                break
            else:
                decoded_words.append(self.converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        accuracy = n_correct / float(n_total)
        cer = distance(''.join(decoded_words), gt_text[0])/len(gt_text[0])
        wer = get_word_error_rate(gt_text[0].split(' '), ''.join(decoded_words).split(' '))

        return loss.item(), accuracy, cer, wer

    def save(self, save_policy='best'):
        print(f"Saving trainer to {self.save_folder}.")
        if len(self.save_folder) > 0 and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            
        if save_policy == 'best':
            if self.model is not None:
                torch.save(self.model.state_dict(), os.path.join(self.save_folder, "model_state_dict"))
                torch.save(self.model, os.path.join(self.save_folder, "model"))
            else:
                torch.save(self.encoder.state_dict(), os.path.join(self.save_folder, "encoder_state_dict"))
                torch.save(self.encoder, os.path.join(self.save_folder, "encoder"))
                torch.save(self.decoder.state_dict(), os.path.join(self.save_folder, "decoder_state_dict"))
                torch.save(self.decoder, os.path.join(self.save_folder, "decoder"))
        elif save_policy == 'last':
            if self.model is not None:
                torch.save(self.model.state_dict(), os.path.join(self.save_folder, "model_last_state_dict"))
                torch.save(self.model, os.path.join(self.save_folder, "model_last"))
            else:
                torch.save(self.encoder.state_dict(), os.path.join(self.save_folder, "encoder_last_state_dict"))
                torch.save(self.encoder, os.path.join(self.save_folder, "encoder_last"))
                torch.save(self.decoder.state_dict(), os.path.join(self.save_folder, "decoder_last_state_dict"))
                torch.save(self.decoder, os.path.join(self.save_folder, "decoder_last"))

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
            if trainer.model is not None:
                trainer.model = torch.load(os.path.join(load_folder, "model"))
            else:
                trainer.encoder = torch.load(os.path.join(load_folder, "encoder"))
                trainer.decoder = torch.load(os.path.join(load_folder, "decoder"))
        elif load_policy == 'last':
            if trainer.model is not None:
                trainer.model = torch.load(os.path.join(load_folder, "model_last"))
            else:
                trainer.encoder = torch.load(os.path.join(load_folder, "encoder_last"))
                trainer.decoder = torch.load(os.path.join(load_folder, "decoder_last"))
        return trainer
