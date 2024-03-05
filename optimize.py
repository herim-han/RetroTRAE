import argparse
from subprocess import run, PIPE
import time  
from contextlib import contextmanager
import pickle

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles
import pandas as pd
import torch 
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, TensorDataset    
import sentencepiece as spm
from pytorch_lightning import Trainer

from data import CustomDataset
from get_property_par import get_property_qvina
from botorch_func import tell, ask
from data import  sos_id, eos_id, pad_id, calculate_prop
from elem_ais import encode as to_ais
from elem_ais import decode as to_smi
from elem_ais import smiles_tokenizer 

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
        sp = spm.SentencePieceProcessor(model_file= f"{args.sp_model}")
        vocab_size = sp.GetPieceSize()
        print(f'vocab size : {vocab_size}')
        from gru_model import MyModel, PositionalEncoder
        #from gru_model_opt import Encoder

        model = MyModel(
                        vocab_size = vocab_size, 
                        d_model = args.d_model,
                        n_layer = args.n_layer,
                        seq_len    = args.seq_len, 
                       )

        model = MyModel.load_from_checkpoint(args.ckpt_path, strict=False)

###################revised code
#        from collections import OrderedDict
#    
#        ckpt = torch.load(args.ckpt_path, map_location=model.device)
#
#        #print(ckpt['state_dict'].keys())
#    
#        pretrained_dict = ckpt['state_dict']
#        new_model_dict = model.state_dict()
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
#        new_model_dict.update(pretrained_dict)
#        model.load_state_dict(new_model_dict)
#    
#        for name, param in ckpt['state_dict'].items():
#            print(name)
#            if name not in model.state_dict().keys():
#                continue
#            if isinstance(param, torch.nn.parameter.Parameter):
#                param = param.data
#            try:
#                model.state_dict()[name].copy_(param)
#                print(model.state_dict()[name].copy_(param).size())
#                print(name, ' copied')
#            except:
#                continue
#
#        exit(-1)

        trainer = Trainer(accelerator=args.accelerator, 
                          devices    =args.devices, 
                          strategy   =args.strategy,
                          max_epochs =args.max_epochs,
                         )

    with timer('Initial Tokens') :
        #initial_tokens = [ to_ais( s )   if args.tokenize_method !='smi' else  " ".join(smiles_tokenizer(s))   for s in initial_smi ]
        initial_tokens = [ to_ais( s , args.sp_model)   if args.tokenize_method !='smi' else  " ".join(smiles_tokenizer(s))   for s in initial_smi ]
        print('encoding ais:\n', initial_tokens)
        #exit(-1)
        initial_tokens = [ sp.encode_as_ids( s ) for s in initial_tokens ]
        print(initial_tokens)
        
#    with timer('Fine Tuning'):
#        model.lr = args.lr
#        initial_data = CustomDataset(initial_tokens, 0.0)
#        #model.encoder.positional_encoder = PositionalEncoder(args.seq_len, model.encoder.positional_encoder.d_model)
#        trainer.fit(model, DataLoader(initial_data, batch_size=args.batch_size) )
    
#    with timer('Debugging') :
#        # for debug
#        print( trainer.validate(model, dataloaders = DataLoader(initial_data, batch_size=args.batch_size) )  )

    with timer('Initial Encoder Pred') :
        model.encoder.positional_encoder = PositionalEncoder(args.seq_len, model.encoder.positional_encoder.d_model)
        initial_feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(initial_tokens, initial_smi,args.batch_size) )
        initial_feature = torch.cat(initial_feature) #1024

    device, dtype = initial_feature.device, initial_feature.dtype
    feature_size = initial_feature.size(-1)
    with timer('Initial Get Property') :
        valid_struct_id=0
        _, initial_docking, initial_SA, _, _, failed_smiles = get_property_qvina(initial_smi, n_repeat = args.n_repeat_docking, num_smiles = valid_struct_id, target=args.target)
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
    for i_iter in range(1, args.opt_iter+1):
        print(f"{i_iter}  ---------------------- optimization input")
        with timer(f'[{i_iter}] Tell'):
            acqf, bounds = tell( 
                                torch.cat( [ item["feature"] for key, item in dict_output.items() if item['feature'] is not None ]), # feature 
                                torch.tensor( sum( [ item["obj_val"] for key, item in dict_output.items() if item['obj_val'] is not None ],[]), device=device, dtype=dtype)
                               )
    
        with timer(f'[{i_iter}] Ask') :
            new_feature = ask(args.num_ask, acqf, bounds, 
                              torch.Tensor([0, args.opt_bound], 
                                           device=device ).unsqueeze(-1).repeat(1, feature_size ) )

#        elif args.opt_method == 'skopt':
#            from skopt import Optimizer
#            from skopt.space import Real
#            new_feature = torch.cat ([item["feature"] for key, item in dict_output.items() if item['feature'] is not None] )
#            #print('!!!!!!!!!!!!!!!!!!!!!!!!!! skopt encoded vectors: ', new_feature.size(), type(new_feature))
#            list_obj     = sum([item["obj_val"] for key, item in dict_output.items() if item['obj_val'] is not None], [])
#            
#            max_vals = torch.max(new_feature, dim=0)[0]
#            min_vals = torch.min(new_feature, dim=0)[0]
#            std_vals = torch.std(new_feature, dim=0)
#        
#            std_vals[std_vals<0.02] = 0.02
#        
#            dimension = [Real(min_vals[i].item()-std_vals[i].item()*args.search_const, max_vals[i].item()+std_vals[i].item()*args.search_const) for i in range(args.d_model*2) ]
#            optimizer = Optimizer(dimension)
#            st = time.time()
#            list_x  = new_feature.detach().cpu().numpy().tolist()
#            optimizer.tell(list_x, list_obj)
#            et = time.time()
#            print(f'time for skopt.tell {et-st:2.3f}')
#            
#            st = time.time()
#            new_list_x = optimizer.ask(args.num_ask)
#            new_feature = torch.tensor(new_list_x, device=new_feature.device, dtype=new_feature.dtype)
#            et = time.time()
#            print(f'time for skopt.ask {et-st:2.3f}')

        with timer(f'[{i_iter}] Decoding') :
            # do decoding
            #print('(decoding) positional seq len', model.encoder.positional_encoder.seq_len)
            tokens = trainer.predict(model, dataloaders=DataLoader( TensorDataset(new_feature) , batch_size=args.batch_size)  )
            del new_feature 
            tokens = torch.cat(tokens)

        with timer(f'[{i_iter}] Generate String') :
            # get string
            print('trainer predict token: ', tokens) 
            list_smi = [ sp.decode_ids(token.detach().cpu().numpy().tolist()) for token in tokens ]
            print('decode_ids (AIS+SMI): ', list_smi)
            smi    = [ to_smi(token) for token in list_smi ] if args.tokenize_method != 'smi' else [ token.replace(" ", "") for token in list_smi ]
        print('!!!!!!!!! decoding generate string', smi)
        total_smi = len(smi)

        # validate the generated smi
        list_mol = [ Chem.MolFromSmiles(s.strip()) for s in smi ]
        valid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if (mol is not None and mol.GetNumAtoms()!=0) ] )
        invalid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if ( mol is None or mol.GetNumAtoms()==0 ) ] )

        if(len(valid_mol)==0): 
            dict_output[i_iter] = { "tokens":None, "smi": None, "feature": None, "docking": None, "SA": None, "obj_val": None, "length":0 } 
            print("It doesn't exist new generated smi after filtering gen smi (by MolFromSmiles, GetNumAtoms)")
            dict_invalid[i_iter] = {
                            "invalid_docking": None,
                            "invalid_decoding": smi
                            }
            continue

        print(f'num_ask: {args.num_ask}, valid_idx: {len(valid_mol)}')
        tokens  = tokens[valid_mol] #type(tokens) = tensor
        valid_smi     = [ smi[idx] for idx in valid_mol ] #type(smi) = list

        print(f'New generated smi: , {valid_smi} \n invalid smi: {total_smi - len(valid_smi)}' )

        with timer(f'[{i_iter}] Get Property') :
            valid_smiles, list_docking, list_SA, success_indices, valid_struct_id, failed_smiles = get_property_qvina(valid_smi, n_repeat = args.n_repeat_docking, num_smiles = valid_struct_id , target=args.target)
        list_score = [ obj_func(docking, SA) for docking, SA in zip(list_docking, list_SA) ]

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

        success_indices = torch.tensor(success_indices, dtype=torch.long)

        with timer(f'[{i_iter}] Encoding') :
            # do encoding
            feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(tokens[success_indices], args.batch_size) )
            feature = torch.cat(feature)

        dict_output[i_iter] = { "tokens": tokens[success_indices], "smi": valid_smiles, "feature": feature, 
                                "docking":list_docking, "SA": list_SA, "obj_val": list_score, "length": len(success_indices) }
        print( dict_output[i_iter]['length'], dict_output[i_iter]["smi"], dict_output[i_iter]["obj_val"] )
    
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
    parser.add_argument( '--tokenize_method', type=str,  choices=['ais', 'smi'], default='ais' )

    # pytorch_lightning Trainer
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default='auto')


    parser.add_argument('--seq_len', help="sequence length", type=int, default=256)

    # model parameters
    parser.add_argument( '--d_model', type=int, default=512 )
    parser.add_argument( '--n_layer', type=int, default=4 )
    parser.add_argument( '--trained_model', type=str, default="lstm", choices =['gru', 'lstm'] )

    # optimize
    parser.add_argument( '--lr', type=float, default=5e-6 )
    parser.add_argument( '--batch_size', type=int, default=1024 )
    parser.add_argument( '--max_epochs', type=int, default=1000 )

    # optimization
    parser.add_argument('--n_repeat_docking', help="number of repeat for docking simulation", type=int, default=10)
    parser.add_argument('--target', type=str, default='5ht1b')
    parser.add_argument( '--search_const', type=float, default=1.5 )
    parser.add_argument('--opt_bound', type=float, default=1.5)
    parser.add_argument('--input_file', type=str, default='pdk4_5' )
    parser.add_argument('--csv_path', type=str, default='./tmp' )

    # Bayesian parameters
    parser.add_argument("--num_ask", type=int, default=100)
    parser.add_argument("--prop", type=int, default=400)

    args = parser.parse_args()  
    assert args.opt_bound>0

    import os
    if not os.path.isdir(args.csv_path):
        os.mkdir(args.csv_path)

    initial_smi = [ line.strip() for line in open(args.input_file).readlines() ]
    print('initial smi:\n',initial_smi)
    result = optimize(args, initial_smi)
