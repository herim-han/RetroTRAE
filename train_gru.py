import os 
import argparse 
import re 
import pickle

import sentencepiece as spm
from tqdm import trange, tqdm

import torch 
from torch import nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader 
torch.set_float32_matmul_precision('medium')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import pad_id, sos_id, eos_id, unk_id
from data import CustomDataset
from gru_model import MyModel
        
def train(args):
    sp = spm.SentencePieceProcessor(model_file= f"{args.sp_model}")
    vocab_size = sp.GetPieceSize()
    print('vocab_size loaded from sp model: ', vocab_size) 

    if args.train_dat is None:
        with open(args.train_filename) as f:
            lines = f.readlines()
        list_tokens=[]
        for line in tqdm( lines[:int(len(lines) * args.train_ratio)], desc='loading data'):#list
            list_tokens.append( [ sp.EncodeAsIds(token.strip())[0] for token in re.split("\s+", line.strip()) ] )

        with open('train.dat', 'wb') as f:
            pickle.dump(list_tokens, f)
    else:
        # load train dataset
        print(f'loading train data from {args.train_dat}')
        with open(args.train_dat, 'rb') as f:
            list_tokens = pickle.load(f)
    
    train_dataset = CustomDataset(list_tokens, '/data/shchoi/herim/RetroTRAE/data/src/train_smi_mw', args.masking_ratio)
    #train_dataset = CustomDataset(list_tokens, '/data/shchoi/herim/RetroTRAE/data/src/train_smi_mw_tmp', args.masking_ratio)
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size,
                               pin_memory=True, num_workers=30, shuffle=True
                              )

    if args.valid_dat is None:
        # construct valid dataset
        with open(args.valid_filename) as f:
            lines = f.readlines()
        list_tokens=[]
        for line in tqdm( lines[:int(len(lines) * args.valid_ratio)], desc='loading data'):#list
            list_tokens.append( [ sp.EncodeAsIds(token.strip())[0] for token in re.split("\s+", line.strip()) ] )

        with open('valid.dat', 'wb') as f:
            pickle.dump(list_tokens, f)
    else:
        # load valid dataset
        print(f'loading valid data from {args.valid_dat}')
        with open(args.valid_dat, 'rb') as f:
            list_tokens = pickle.load(f)

    valid_dataset = CustomDataset(list_tokens, '/data/shchoi/herim/RetroTRAE/data/src/valid_smi_mw') # no masking 
    #valid_dataset = CustomDataset(list_tokens, '/data/shchoi/herim/RetroTRAE/data/src/valid_smi_mw_tmp') # no masking 
    valid_loader  = DataLoader(valid_dataset, batch_size = args.batch_size,
                               pin_memory=True, num_workers=30, shuffle=False, 
                              )

    model = MyModel(vocab_size = vocab_size, 
                    seq_len = max(train_dataset.seq_len, valid_dataset.seq_len), 
                    d_model = args.d_model,
                    dropout=args.dropout, 
                    n_layer = args.n_layer,
                    lr = args.lr)

    if args.ckpt_path is not None:
        model = MyModel.load_from_checkpoint(args.ckpt_path, map_location=model.device)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{valid_acc:.3f}', 
                                          verbose=True, 
                                          save_top_k=-1, 
                                          monitor='valid_acc', 
                                          mode='max',
                                          every_n_epochs=args.check_val_every_n_epoch)

    trainer = Trainer(accelerator=args.accelerator, 
                      devices    =args.devices, 
                      strategy   =args.strategy,
                      max_epochs =args.max_epochs,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      callbacks = [checkpoint_callback],
                     )

####training
    trainer.fit(model, train_loader, valid_loader, ckpt_path = args.ckpt_path)
####validation
    #trainer.validate(model, valid_loader, ckpt_path = args.ckpt_path)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # restart
    parser.add_argument('--ckpt_path', help="dataset path", type=str, default = None)    
    # data
    parser.add_argument( '--train_filename', type=str )
    parser.add_argument( '--valid_filename', type=str )
    parser.add_argument( '--train_dat', type=str, default=None )
    parser.add_argument( '--valid_dat', type=str, default=None )
    parser.add_argument( '--train_ratio', type=float, default=1.0 )
    parser.add_argument( '--valid_ratio', type=float, default=1.0 )
    parser.add_argument( '--masking_ratio', type=float, default=0.0 )

    # sentencepiece
    parser.add_argument( '--sp_model', type=str )

    # optimize
    parser.add_argument( '--lr', type=float, default=5e-4 )
    parser.add_argument( '--batch_size', type=int, default=1024 )
    parser.add_argument( '--max_epochs', type=int, default=1 )

    # model parameters
    parser.add_argument( '--dropout', type=float, default=0.1 )
    parser.add_argument( '--d_model', type=int, default=512 )
    parser.add_argument( '--n_layer', type=int, default=4 )

    # pytorch_lightning Trainer
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default='auto')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1 )


    args = parser.parse_args()  

    assert args.train_ratio <=1.0
    train(args)
