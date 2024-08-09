from torch import nn

import torch
import math
from data import  sos_id, eos_id, pad_id

import pytorch_lightning as pl

class MyModel2(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1, lr = 1e-4, kl_factor=0.03, comp_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(vocab_size, seq_len, d_model, n_head, n_layer, dropout, comp_size)
        self.decoder = Decoder(vocab_size, seq_len, d_model, n_head, n_layer, dropout, comp_size)

        self.loss_f  = nn.NLLLoss(ignore_index=0)
        self.lr = lr

        self.n_layer = n_layer
        self.seq_len = seq_len
        self.list_test_output=[]

        self.kl_loss = lambda mu, logvar:  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())* kl_factor
    def training_step(self, batch, batch_idx):
        src, trg, prop = batch
        src_inp, mu, logvar, properties = self.encoder(src, src ==pad_id , prop)#compressed dim
        loss2 = self.kl_loss(mu, logvar)
        #print('Encoder (Compressed dim): ', mu.size(), logvar.size() )

        feature = self.decoder.sampling(mu, logvar)#recover dim
        feature = torch.cat((feature, properties), 1)
        #print('Decoder (Recover dim): ', feature.size() )
        output = self.decoder( src_inp, feature )

        loss1 = self.loss_f(output[:,:-1].reshape(-1, output.size(-1)) , trg[:,1:].reshape(-1) )
        loss = loss1+loss2
        self.log( "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        self.log( "train_loss1", loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        self.log( "train_loss2", loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size =  src.size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg, prop = batch
        src_inp, mu, logvar, properties = self.encoder(src, src ==0, prop )#
        #print('Encoder (Compressed dim): ', mu.size(), logvar.size() )

        feature = self.decoder.sampling(mu, logvar)
        feature = torch.cat((feature, properties), 1)
        #print('Decoder (Recover dim): ', feature.size() )
        ## generate probability
        output = self.decoder(src_inp, feature)   # (B, L, d_model)
        # generate tokens 
        val_tokens = self.generate_tokens(feature, seq_len=src_inp.size(1) )

        loss1 = self.loss_f(output[:,:-1].reshape(-1, output.size(-1)) , trg[:,1:].reshape(-1) )
        loss2= self.kl_loss(mu, logvar)
        loss = loss1+loss2
        self.log( "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )
        self.log( "valid_loss1", loss1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0) , sync_dist=True)
        self.log( "valid_loss2", loss2, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )
        
        trg = trg*torch.cumprod( trg!=eos_id, dim=-1)
        val_tokens = val_tokens*torch.cumprod( val_tokens !=eos_id, dim=-1)
        print(torch.all(trg == val_tokens, dim=-1))
        acc = torch.all(trg == val_tokens, dim=-1).sum() / src.size(0)

        self.log("valid_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size = src.size(0), sync_dist=True )

    def predict_step(self, batch, batch_idx):
        #print('!!!!!!!!!!!!  MyModel.predict_step')
        init_feature = batch[0]
        mw_prop = batch[1]
        mu, logvar = torch.split(init_feature, init_feature.size(-1)//2, dim=-1 )
        feature = self.decoder.sampling(mu, logvar)

        mw_mean = torch.tensor([372.53]).to(feature.device)
        mw_std = torch.tensor([88.89]).to(feature.device)

#        properties = torch.stack([(mw_prop-mw_mean)/mw_std for i in range(feature.size(0))]).to(torch.float32)
        properties = ((mw_prop-mw_mean)/mw_std).clone().detach().unsqueeze(-1).to(torch.float32)
        feature = torch.cat((feature, properties), dim=-1)
        tokens = self.generate_tokens(feature)
        return tokens

    def generate_tokens(self, feature, seq_len=0):
        #print('!!!!!!!!!! MyModel.generate_tokens')
        if len(feature.size())==2:
            feature = feature.unsqueeze(0).repeat(self.n_layer, 1, 1)
        elif len(feature.size())==3:
            pass
        else:
            assert RuntimeError ('Size of feature should be (batch_size, d_model) or (n_layer, batch_size, d_model) ')  
        seq_len = self.seq_len if seq_len==0 else seq_len
        output = torch.tensor( [sos_id], dtype=torch.long, device=feature.device).view(1,1).repeat(feature.size(1),1) # (B, 1) 

        for _ in range(1, seq_len):
            embed_tokens = self.encoder.src_embedding(output) # (B, L) => (B, L, d_model)
            embed_tokens = self.encoder.positional_encoder(embed_tokens) # (B, L, d_model) => (B, L, d_model)
                            # decode_single (B, vocab_size)
            new_token    = self.decoder.decode_single(embed_tokens, feature ).topk(1, dim=-1)[1]
            output       = torch.cat( [ output, new_token.view(-1, 1)], dim=1)

        mask = torch.cat ( [ torch.ones((output.size(0), 1), device=output.device, dtype=output.dtype ), 
                             torch.cumprod( output!=eos_id, dim=-1)[:,:-1] ], dim = -1 ) 

        output = output*mask
        
        return output
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Encoder(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1, comp_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.src_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(seq_len, d_model)
        self.encoder = nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True), n_layer)

        self.readout1 = nn.Linear(d_model+1, d_model)#added prop layer
        self.latent_mu = nn.Linear(d_model+1, comp_size)#added prop layer
        self.readout2 = nn.Linear(d_model+1, d_model)#added prop layer
        self.latent_logvar = nn.Linear(d_model+1, comp_size)#added prop layer

    def forward(self, src, e_mask=None, prop=None):
        #print('!!!!!!!!!!!! self.encoder.forward')
        src_inp = self.src_embedding(src) # (B, L) => (B, L, d_model)
        src_inp = self.positional_encoder(src_inp) # (B, L, d_model) => (B, L, d_model)
        e_output = self.encoder(src_inp, src_key_padding_mask=e_mask) # (B, L, d_model)
        mw_mean = torch.tensor([372.53]).to(src_inp.device)
        mw_std = torch.tensor([88.89]).to(src_inp.device)
        properties = ((prop-mw_mean)/mw_std).clone().detach().unsqueeze(-1).to(torch.float32)
        
        mu     = self.readout1(torch.cat((e_output[:,0], properties), 1))
        mu     = self.latent_mu(torch.cat((mu, properties), 1) )
        logvar = self.readout2(torch.cat((e_output[:,0], properties), 1))
        logvar = self.latent_logvar(torch.cat((logvar, properties), 1) )
        return src_inp, mu, logvar, properties # (B, n_layer, d_model)

    def predict_step(self, batch, batch_idx):
        print('!!!!!!!!!!! model.Encoder.predict_step')
        #exit(-1)
        src, e_mask, prop = batch
        #print('token idx', src, src.size() )
        #print('prop in Encoder', prop)
        src_inp, mu, logvar, properties = self.forward(src, e_mask, prop)
        return torch.cat([mu, logvar], dim=-1)

class Decoder(pl.LightningModule):
    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1, comp_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        #self.decoder = nn.GRU( input_size=d_model, hidden_size=d_model, num_layers=n_layer, dropout=dropout, batch_first=True)
        self.decoder = nn.GRU( input_size=d_model, hidden_size=d_model+1, num_layers=n_layer, dropout=dropout, batch_first=True)
        self.recover_layer = nn.Sequential(nn.Linear(comp_size, d_model))
        self.output_linear = nn.Linear(d_model+1, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.d_model = d_model
        self.n_layer = n_layer

    def forward(self, src_inp, feature):
        #print('!!!!!!!!!!! MyModel.decoder.forward')
        if len(feature.size())==2:
            feature = feature.unsqueeze(0).repeat(self.n_layer, 1, 1)
        elif len(feature.size())==3:
            pass
        else:
            assert RuntimeError ('Size of feature should be (batch_size, d_model) or (n_layer, batch_size, d_model) ')
        d_output, _ = self.decoder(src_inp, feature) # (B, L, d_model+1)

        output = self.softmax( self.output_linear(d_output) ) # (B, L, d_model+1) => # (B, L, trg_vocab_size) 
        return output

    def decode_single(self, src_inp, feature ):
        # _h (n_layer, B, d_model)
        assert feature.size(0) == self.n_layer

        d_output = self.decoder(src_inp, feature)[0] # (B, L, d_model)
        d_output =self.softmax( self.output_linear(d_output[:,-1]) ) # (B, d_model) => # (B, trg_vocab_size) 
        return d_output

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var) #(B, #, comp_size)
        eps = torch.randn_like(std)
        #return eps.mul(std).add(mu)
        return self.recover_layer(eps.mul(std).add(mu) )

#class Model(nn.Module):
#    def __init__(self, vocab_size, seq_len, d_model=512, n_head=8, n_layer=4, dropout =0.1):
#        super().__init__()
#        self.vocab_size = vocab_size
#
#        self.src_embedding = nn.Embedding(self.vocab_size, d_model)
#        self.positional_encoder = PositionalEncoder(seq_len, d_model)
#        self.encoder = nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True), n_layer)
#        self.decoder = nn.GRU( input_size=d_model, hidden_size=d_model, num_layers=n_layer, dropout=dropout, batch_first=True)
#        self.output_linear = nn.Linear(d_model, self.vocab_size)
#        self.softmax = nn.LogSoftmax(dim=-1)
#
#        self.seq_len = seq_len
#
#        self.d_model = d_model
#        self.n_layer = n_layer
#        self.n_head  = n_head
#
#    def forward(self, src_input, e_mask=None):
#        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
#        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
#
#        e_output = self.encoder(src_input, src_key_padding_mask=e_mask) # (B, L, d_model)
#        _h = e_output[:,0].unsqueeze(0).repeat(self.n_layer,1,1) # (B, n_layer, d_model)
#
#        d_output, _h = self.decoder(src_input, _h) # (B, L, d_model)
#        
#        output = self.softmax( self.output_linear(d_output) ) # (B, L, d_model) => # (B, L, trg_vocab_size) 
#
#        return output
#
#    def validation(self, src_tokens, e_mask=None):
#        src_input = self.src_embedding(src_tokens) # (B, L) => (B, L, d_model)
#        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
#
#        e_output = self.encoder(src_input, src_key_padding_mask=e_mask) # (B, L, d_model)
#        _h = e_output[:,0].unsqueeze(0).repeat(self.n_layer,1, 1) # (n_layer, B, d_model)
#
#
#        # generate probability
#        d_output = self.decoder(src_input, _h)[0] # (B, L, d_model)
#        prob = self.softmax( self.output_linear(d_output) ) # (B, L, d_model) => # (B, L, trg_vocab_size) 
#
#        # generate tokens from greedy search: what is self.decode??
#        return prob, self.decode(_h, seq_len = src_tokens.size(1) )
#
#
#    def decode(self, _h, seq_len=0):
#        if len(_h.size())==2:
#            _h.unsqueeze(0).repeat(self.n_layer, 1, 1)
#        elif len(_h.size())==3:
#            pass
#        else:
#            assert RuntimeError ('Size of _h should be (n_layer, batch_size, d_model) ')  
#        if seq_len==0:
#            seq_len = self.seq_len
#                            
#        # generate tokens from greedy search
#        output = torch.tensor( [sos_id], dtype=torch.long, device=_h.device).view(1,1).repeat(_h.size(1),1) # (B, 1) 
#        for _ in range(1, seq_len):
#            embed_tokens = self.src_embedding(output) # (B, L) => (B, L, d_model)
#            embed_tokens = self.positional_encoder(embed_tokens) # (B, L, d_model) => (B, L, d_model)
#            new_token = self.decode_single(_h, embed_tokens).topk(1, dim=-1)[1]
#            output = torch.cat( [ output, new_token.view(-1, 1)], dim=1)
#        return output
#        
#    def decode_single(self, _h, src_input):
#        # _h (n_layer, B, d_model)
#        assert _h.size(0) == self.n_layer
#
##        if src_input is None:
##            src_tokens = torch.LongTensor( [sos_id], device=_h.device).reshape(1,-1).repeat(_h.size(0),1)
##            src_input = self.src_embedding(src_tokens) # (B, L) => (B, L, d_model)
##            src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
##            del src_tokens
#
#        d_output = self.decoder(src_input, _h)[0] # (B, L, d_model)
#        return self.softmax( self.output_linear(d_output[:,-1]) ) # (B, d_model) => # (B, trg_vocab_size) 
        

class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        # Make initial positional encoding matrix with 0
        self.positional_encoding= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    self.positional_encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    self.positional_encoding[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        self.positional_encoding = self.positional_encoding.unsqueeze(0) # (1, L, d_model)
        #self.positional_encoding = self.positional_encoding.to(device=device).requires_grad_(False)
        self.positional_encoding = torch.nn.parameter.Parameter(self.positional_encoding, requires_grad=False)
        self.d_model = d_model

    def forward(self, x):
        #print(x.size(1), self.positional_encoding.size())
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding[:,:x.size(1)] # (B, L, d_model)
        #.to(x.device) # (B, L, d_model)
        #x = x + self.positional_encoding # (B, L, d_model)

        return x
