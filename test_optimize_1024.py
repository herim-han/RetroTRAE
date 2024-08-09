import argparse
from subprocess import run, PIPE
import time  
from contextlib import contextmanager
import pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles, QED
import pandas as pd
import torch 
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, TensorDataset    
import sentencepiece as spm
from pytorch_lightning import Trainer
from data import CustomDataset, sos_id, eos_id, pad_id, calculate_prop
from get_property_par import get_property_qvina
from botorch_func import tell, ask
from elem_ais import encode as to_ais
from elem_ais import decode as to_smi
from elem_ais import smiles_tokenizer 
import selfies as sf
import torch.nn as nn

dict_timer = {}

@contextmanager
def timer(desc=""):
    st = time.time()
    yield 
    et = time.time()
    elapsed_time = et-st
    if desc in dict_timer.keys() :
        dict_timer[desc].append(elapsed_time)
    else:
        dict_timer[desc] = [elapsed_time]        
    print( f"Elapsed Time[{desc}]: ", elapsed_time )

def get_encoder_dataloader( tokens, smiles,  batch_size=1024 ):
    """ 
    """
    from torch.utils.data import dataloader, TensorDataset  
    from torch.nn.utils.rnn import pad_sequence
    tokens = pad_sequence([ torch.tensor(item, dtype=torch.long) for item in tokens] , batch_first=True, padding_value=pad_id) #set max_length
    seq_len = torch.sum( torch.any( tokens != pad_id, dim=0 ) )
    tokens= tokens[:, :seq_len] #torch.Tensor(n, l)
    e_mask = (tokens == pad_id )
    prop = torch.Tensor([args.prop]).repeat(tokens.size(0))
    
    return DataLoader( TensorDataset(tokens, e_mask, prop), batch_size=batch_size)

def optimize(args, initial_smi, obj_func = lambda docking, SA: -docking-0.5*SA*SA ):
    with timer('Load Model ') :
        print('sp model', args.sp_model)
        print('load training model', args.ckpt_path)
        sp = spm.SentencePieceProcessor(model_file= f"{args.sp_model}")
        vocab_size = sp.GetPieceSize()
        print(f'vocab size : {vocab_size}')

        from gru_model import MyModel, PositionalEncoder

        model = MyModel(
                         vocab_size = vocab_size,
                         d_model = args.d_model,
                         n_layer = args.n_layer,
                         seq_len = args.seq_len,
                         lr = args.lr
                        )

        device = torch.device('cuda') if torch.cuda.is_available()==True else torch.device('cpu')

        model = MyModel.load_from_checkpoint(args.ckpt_path, strict=False, map_location=device)
        print('device', device)

#        print('current cuda device', torch.cuda.current_device())
#        print('count GPUS', torch.cuda.device_count())
#        model = MyModel2.load_from_checkpoint(args.ckpt_path, strict=False, map_location=model.device)

        trainer = Trainer(accelerator=args.accelerator, 
                          devices    =args.devices, 
                          strategy   =args.strategy,
                          max_epochs =args.max_epochs
                         )
        print('model loaded: ', model.device)
        print('torch.cuda.is_available ', torch.cuda.is_available() )
#        ckpt = torch.load(args.ckpt_path, map_location=model.device)
#
###################revised code
#        from collections import OrderedDict
#        
#        pretrained_dict = ckpt['state_dict']
#        new_model_dict = model.state_dict()
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
#        new_model_dict.update(pretrained_dict)
#        model.load_state_dict(new_model_dict)
    
#        for name, param in ckpt['state_dict'].items():
#            #print(name)
#            if name not in model.state_dict().keys():
#                continue
#            if isinstance(param, torch.nn.parameter.Parameter):
#                param = param.data
#            try:
#                model.state_dict()[name].copy_(param)
#                #print(model.state_dict()[name].copy_(param).size())
#                #print(name, '\n ', ' copied!!!')
#            except:
#                print(name, ' none copied!')
#                continue


    with timer('Initial Tokens') :
        initial_tokens =[
                          to_ais(s, args.sp_model) if args.tokenize_method =='ais'
                          else " ".join(smiles_tokenizer(sf.encoder(s))) if args.tokenize_method=='selfies'
                          else " ".join(smiles_tokenizer(s)) for s in initial_smi ]

        print('encoding ais:\n', initial_tokens)
        initial_tokens = [ sp.encode_as_ids( s ) for s in initial_tokens ]
        print(initial_tokens)
        
#    with timer('Fine Tuning'):
#        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
#
#        checkpoint_callback = ModelCheckpoint(filename=f'ft_vocab_{vocab_size}'+'_{epoch}_{valid_acc:.3f}',
#                                              dirpath='./cvae_e20_ckpt_file/fine-tuning', 
#                                              verbose=True, 
#                                              save_top_k=-1, 
#                                              monitor='valid_acc', 
#                                              mode='max',
#                                              every_n_epochs=1000,
#                                              )
#
#        trainer = Trainer(accelerator=args.accelerator, 
#                          devices    =args.devices, 
#                          strategy   =args.strategy,
#                          max_epochs =args.max_epochs,
#                          check_val_every_n_epoch=args.check_every_n_epoch,
#                          callbacks = [checkpoint_callback],
#                         )
#
#        model.lr = args.lr
#        initial_mw = [QED.properties(Chem.MolFromSmiles(smi)).MW for smi in initial_smi ]
#        with open('ft_mw', 'w') as f:
#            for smi in initial_smi:
#                mw = QED.properties(MolFromSmiles(smi)).MW
#                f.write(str(mw)+'\n')
#        initial_data = CustomDataset(initial_tokens, 'ft_mw', 0.0)
#        dataloader = DataLoader(initial_data, batch_size = args.batch_size)
#        #model.encoder.positional_encoder = PositionalEncoder(args.seq_len, model.encoder.positional_encoder.d_model)
#        trainer.fit(model, dataloader, dataloader) #(model, train_data, valid_data)
#        print(':::::::::End Fine Tuning:::::::')
#   
#    with timer('Debugging') :
#        # for debug
#        print( trainer.validate(model, dataloaders = DataLoader(initial_data, batch_size=args.batch_size) )  )

    with timer('Initial Encoder Pred') :
        model.encoder.positional_encoder = PositionalEncoder(args.seq_len, model.encoder.positional_encoder.d_model)
        initial_feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(initial_tokens, initial_smi,args.batch_size) )#Encoder.predict_step
        initial_feature = torch.cat(initial_feature) #1024
        print('Feature size (Compressed dim): ', initial_feature.size())

    device, dtype = initial_feature.device, initial_feature.dtype
    feature_size = initial_feature.size(-1)

    with timer('Initial Get Property') :
        valid_struct_id=0
        _, initial_docking, initial_SA, _, _, failed_smiles = get_property_qvina(initial_smi, n_repeat = args.n_repeat_docking, csv_path=f'{args.csv_path}/tmp', num_smiles = valid_struct_id, target=args.target)
    if len(failed_smiles)>0:
        print('Error! get_property return error for initial smiles. please check initial smiles')
        return

    dict_invalid = {0: {
                        "invalid_docking": failed_smiles,
                        "invalid_decoding": None
                        }
                   }

    print("initial docking: ", initial_docking)        
    print("initial SA: ", initial_SA)
    dict_output = {0: { "tokens": initial_tokens, 
                        "smi": initial_smi, 
                        "feature": initial_feature, 
                        "docking": initial_docking, 
                        "SA": initial_SA,
                        "obj_val": [ obj_func(docking,SA) for docking, SA in zip(initial_docking, initial_SA) ],
                       }
                  }
    print("initial obj:", dict_output[0]['obj_val' ] )

    all_feature = initial_feature #initialize feature tensor
    obj_val     = torch.tensor([ obj_func(docking,SA) for docking, SA in zip(initial_docking, initial_SA) ], device=device, dtype=dtype)
    for i_iter in range(1, args.opt_iter+1):
        print(f"{i_iter}  ---------------------- optimization input")
        if args.opt_method == 'botorch':
            with timer(f'[{i_iter}] Tell'):
                #feature = torch.cat( [ item["feature"] for key, item in dict_output.items() if item['feature'] is not None ])
                #obj_val = torch.tensor( sum( [ item["obj_val"] for key, item in dict_output.items() if item['obj_val'] is not None ],[]), device=device, dtype=dtype)
                print('!!!!!!!!!!!!!! optimizer feature, val: ', all_feature.size(), obj_val.size() )
                acqf, bounds = tell( all_feature, obj_val )
        
            with timer(f'[{i_iter}] Ask') :
                new_feature = ask(args.num_ask, acqf, bounds, 
                                  torch.Tensor([0, args.opt_bound], 
                                               device=device ).unsqueeze(-1).repeat(1, feature_size ) )
                print('!!!!!!!!!!!new feature: ', new_feature.size())
        
        with timer(f'[{i_iter}] Decoding') :
            # do decoding
            new_prop = torch.Tensor([args.prop]).repeat(new_feature.size(0))
            tokens = trainer.predict(model, dataloaders=DataLoader( TensorDataset(new_feature, new_prop) , batch_size=args.batch_size)  ) #model.predict_step = val_tokens (Recover Dim)
            #del new_feature 
            tokens = torch.cat(tokens)
            print('Decoding tokens: ', tokens.size() )

        with timer(f'[{i_iter}] Generate String') :
            # get string
            list_smi = [ sp.decode_ids(token.detach().cpu().numpy().tolist()) for token in tokens ]
            #0430print('decode_ids (AIS+SMI): ', list_smi)
            smi    = [ to_smi(token) for token in list_smi ] if args.tokenize_method == 'ais' else [ sf.decoder(token.replace(" ", "")) for token in list_smi] if args.tokenize_method == 'selfies' else [ token.replace(" ", "") for token in list_smi ]
        print('!!!!!!!!! decoding generate string', len(smi) )
        total_smi = len(smi)

        # validate the generated smi
        list_mol = [ Chem.MolFromSmiles(s.strip()) for s in smi ]
        valid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if (mol is not None and mol.GetNumAtoms()!=0) ] )#idx
        invalid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if ( mol is None or mol.GetNumAtoms()==0 ) ] )
        
        if(len(valid_mol)==0): 
            dict_output[i_iter] = { "tokens":None, "smi": None, "feature": None, "docking": None, "SA": None, "obj_val": None, "length":0 } 
            print("It doesn't exist new generated smi after filtering gen smi (by MolFromSmiles, GetNumAtoms)")
            dict_invalid[i_iter] = {
                            "invalid_docking": None,
                            "invalid_decoding": smi
                            }
            continue

        print(f'num_ask: {args.num_ask}, valid_mol: {len(valid_mol)}')
        #0430tokens  = tokens[valid_mol] #type(tokens) = tensor
        valid_smi     = [ smi[idx] for idx in valid_mol ] #type(smi) = list

#0430        print(f'New generated smi: , {valid_smi} \n invalid smi: {total_smi - len(valid_smi)}' )
        with timer(f'[{i_iter}] Get Property') :
            valid_smiles, list_docking, list_SA, success_indices, valid_struct_id, failed_smiles = get_property_qvina(valid_smi, n_repeat = args.n_repeat_docking, csv_path = f'{args.csv_path}/tmp', num_smiles = valid_struct_id , target=args.target)
            print('failed docking program', len(valid_mol) - len(success_indices) )

        #print(obj_val, obj_val.size())

        if(len(success_indices)==0): 
            dict_output[i_iter] = { "tokens":None, "smi": None, "feature": None, "docking": None, "SA": None, "obj_val": None, "length":0 } 
            dict_invalid[i_iter] = {
                            "invalid_docking": failed_smiles,
                            "invalid_decoding": None
                            }
            continue

        dict_invalid[i_iter] = {
                            "invalid_docking": failed_smiles,
                            "invalid_decoding": [ smi[idx] for idx in invalid_mol ]
                            }

        success_indices = torch.tensor(success_indices, dtype=torch.long)#by docking

        with timer(f'[{i_iter}] Encoding') :
            # do encoding
            feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(tokens, args.batch_size) )
            feature = torch.cat(feature)
            print('Encoder.predict token size: ', tokens.size() )
            print('valid/success token size: ', tokens[valid_mol][success_indices].size() )

        list_score = [ obj_func(docking, SA) for docking, SA in zip(list_docking, list_SA) ]
        valid_idx = valid_mol[success_indices]
        
        #print(f'valid_idx, score : {len(valid_idx)}, {len(list_score)}, {type(valid_idx)}')
        list_obj_score = [list_score[valid_idx.tolist().index(i)] if i in valid_idx.tolist() else -100 for i in range(args.num_ask)]

        #update feature/obj_val for optimizer
        obj_val = torch.cat((obj_val, torch.tensor(list_obj_score) ), dim=0)
        all_feature = torch.cat((all_feature, new_feature), dim=0 )
        #print('concat all feature: ', all_feature.size() )
        print('invalid_idx (obj_val=-100/feature=False): ', invalid_mol)
        print('obj_val ', obj_val)
        print('compare feature in all_feature ', (all_feature[:,None] == new_feature[valid_mol][success_indices]).all(-1).any(-1))
        
        dict_output[i_iter] = { "tokens": tokens[success_indices], "smi": valid_smiles, "feature": new_feature, 
                                "docking":list_docking, "SA": list_SA, "obj_val": list_score, "length": len(success_indices) }
#        print( dict_output[i_iter]['length'], dict_output[i_iter]["smi"], dict_output[i_iter]["obj_val"] )
    
    with timer(f'save result') :   
        with open(f'{args.csv_path}/optimize_result.pkl', 'wb') as f:
            pickle.dump(dict_output, f)
        with open(f'{args.csv_path}/invalid_result.pkl', 'wb') as f:
            pickle.dump(dict_invalid, f) 

    print("========  timer  ========")            
    print( "\n".join( [ f"{key}: {sum(dict_timer[key])}" for key in dict_timer.keys() ] ) )

    print("======== End opt ========")

    with open(f'{args.csv_path}/optimize_result.pkl', 'rb') as f:
        data = pickle.load(f)
    
    tmp_smi = []
    tmp_dxc = []
    tmp_obj = []
    tmp_sa = []
    tmp_tokens = []
    num_smi = 0
    for i in range(1, args.opt_iter+1):
        try:
            num_smi += len(data[i]['smi'])
            tmp_smi.append(data[i]["smi"])
            tmp_dxc.append(data[i]['docking'])
            tmp_sa.append(data[i]['docking'])
            tmp_obj.append(data[i]['obj_val'])
            tmp_tokens.append(data[i]['tokens'])
        except:
            continue

    list_smi = [item for row in tmp_smi for item in row]
    list_dxc = [item for row in tmp_dxc for item in row]
    list_sa  = [item for row in tmp_sa for item in row]
    list_obj = [item for row in tmp_obj for item in row]
    print(len(list_smi), len(list_dxc), len(list_obj), list_smi)
    print('total generate smi', num_smi)
    print('total uniq smi (remove duplicate)', len(list_smi))
    zinc_DB = [line.strip() for line in open("can_zinc", 'r')]
    binding_DB = [line.strip() for line in open("ocan_pdk4_ligand.txt", 'r')]
    print("novel smiles (not in zinc/bindingDB)", len(list(filter(lambda s: not(s in zinc_DB) and not(s in binding_DB), list_smi))))

    i=0
    for tokens in torch.cat(tmp_tokens):
        t = tokens[1==(tokens!=2).int().cumprod(0)]
        _, counts = torch.unique_consecutive(t, return_counts=True)
        if torch.any(counts >10):
            i+=1

    print('torch consecutive counts > 10', i)
    for i in [3, 5, 10]:
        list_sort_obj = sorted(list_obj, reverse=True)[:i]
        print(f'top-{i} min/max/mean {round(min(list_sort_obj),3)}, {round(max(list_sort_obj),3)}, {round(sum(list_sort_obj)/len(list_sort_obj),3)}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', help="dataset path", type=str, default = None)
    parser.add_argument('--opt_iter', help="number of optimization iterations", type=int, default = 100)
    parser.add_argument('--init_smiles', type=int, default=10 )

    # sentencepiece
    parser.add_argument( '--sp_model', type=str )
    parser.add_argument( '--tokenize_method', type=str,  choices=['ais', 'smi', 'selfies'], default='ais' )

    # pytorch_lightning Trainer
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default='auto')


    parser.add_argument('--seq_len', help="sequence length", type=int, default=256)

    # model parameters
    parser.add_argument( '--d_model', type=int, default=512 )
    parser.add_argument( '--n_layer', type=int, default=4 )
    parser.add_argument( '--comp_size', type=int, default=32 )
    parser.add_argument( '--check_every_n_epoch', type=int, default=5 )

    # optimize
    parser.add_argument( '--lr', type=float, default=5e-6 )
    parser.add_argument( '--batch_size', type=int, default=1024 )
    parser.add_argument( '--max_epochs', type=int, default=2000 )

    # optimization
    parser.add_argument('--n_repeat_docking', help="number of repeat for docking simulation", type=int, default=10)
    parser.add_argument('--target', type=str, default='pdk4')
    parser.add_argument('--search_const', type=float, default=1.5 )
    parser.add_argument('--opt_bound', type=float, default=1.5)
    parser.add_argument('--opt_method', type=str, default='botorch')
    parser.add_argument('--input_file', type=str, default='pdk4_5' )
    parser.add_argument('--csv_path', type=str, default='./tmp' )
    parser.add_argument('--init_num', type=int, default=5 )

    # Bayesian parameters
    parser.add_argument("--num_ask", type=int, default=100)
    parser.add_argument("--prop", type=int, default=400)

    args = parser.parse_args()  
    assert args.opt_bound>0

    import os
    if not os.path.isdir(args.csv_path):
        os.mkdir(args.csv_path)

    initial_smi = [ line.strip() for line in open(args.input_file).readlines() ][:args.init_num]
    print(f'initial smi load from {args.input_file}:\n',initial_smi)
    result = optimize(args, initial_smi)
