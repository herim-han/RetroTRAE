import torch 
from torch.utils.data import Dataset
#from transformer import *
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import DataStructs
import pandas as pd 
import sentencepiece as spm
from elem_ais import decode as to_smi
from multiprocessing import Pool
import numpy as np

pad_id = 0 ; sos_id = 1 ; eos_id = 2 ; unk_id = 3

class CustomDataset(Dataset):
    def __init__(self, list_tokens, filename, masking_ratio=0.0):#pad=0, unk=3
        super().__init__()
        self.seq_len = max( [ len(tokens) for tokens in list_tokens])+2
        self.masking_ratio = masking_ratio
        list_tokens = [ pad_or_truncate([sos_id] + tokens + [eos_id], self.seq_len) for tokens in list_tokens ]
        self.data = torch.LongTensor(list_tokens)
        self.prop = np.loadtxt(filename, dtype=np.float32)

#        self.smi = open(filename).readlines()
#        with Pool(24) as p:
#            list_tokens = ((tokens, self.seq_len) for tokens in list_tokens)
#            list_tokens = list( p.map (pad_or_truncate, list_tokens ) )
#            self.data = torch.LongTensor(list_tokens)
#            self.prop = list (p.map (calculate_prop, self.smi) )

    def __getitem__(self, idx):
        src_data = self.data[idx].clone().detach()
        src_data[torch.rand(src_data.size(), device=src_data.device) < self.masking_ratio ] = unk_id
        src_data[self.data[idx]==pad_id] = pad_id
        prop = torch.tensor(self.prop[idx])
        return src_data, self.data[idx], prop

    def __len__(self):
        return len(self.data)

#def pad_or_truncate(inp):
#    tokenized_text, seq_len = inp[0], inp[1]
#    tokenized_text = [sos_id] + tokenized_text + [eos_id]
def pad_or_truncate(tokenized_text, seq_len):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]
    return tokenized_text

def calculate_prop(s):
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(s))
    prop_mw= QED.properties(mol).MW
    return prop_mw

######
def build_model(sp_vocab):
    print(f"{sp_vocab} vocabulary is building...")
    #print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}
    with open(sp_vocab, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word     # max word length = 79
    return Transformer(vocab_size=len(src_i2w), seq_len=100)

def get_data_loader(train_dataset, batch_size, masking_ratio=0.0, ddp=False, rank=0, world_size=1, num_workers=-1, sp_model=None):##ddp=True > num_workers=0//pin_memory=True
    sp = spm.SentencePieceProcessor()
    sp.Load(f"./data/sp/{sp_model}.model")
    
    tokenized_list=[]
    for line in tqdm(train_dataset):#list
        tokenid_list=[]
        for token in re.split("\s+", line.strip()):
            token_id=sp.EncodeAsIds(token.strip())
            tokenid_list.append(token_id[0])
        tokenized_list.append(pad_or_truncate([sos_id] + tokenid_list + [eos_id]))
    
    dataset = CustomDataset(tokenized_list, masking_ratio)

    if ddp:
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
        dataloader = DataLoader(dataset, batch_size=batch_size//world_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
