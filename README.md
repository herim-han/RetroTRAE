## test example ::
### training_command : ./run.sh
```
python train_gru.py     --sp_model ../data/sp/ais_vocab_25.model \
                        --train_ratio 1.0 \
                        --valid_ratio 1.0 \
                        --max_epochs 300  \
                        --batch_size 512 \
                        --accelerator gpu \
                        --devices 1 \
                        --check_val_every_n_epoch 5 \
                        --strategy='ddp'\
                        --lr  5e-5 \
                        --masking_ratio 0 \
                        --valid_filename ../data/src/ais_vocab_25_valid.txt \
                        --train_filename ../data/src/ais_vocab_25_train.txt 
```

### optimization_command : ./run_opt.sh
```
python optimize.py      --sp_model ../data/sp/ais_vocab_25.model \
                        --ckpt_path cvae_ckpt_file/run_ais_25/epoch=29.ckpt \
                        --csv_path testtest \
                        --n_repeat 1 --num_ask 5 --tokenize_method ais \
                        --strategy='ddp' --opt_iter 2 \
                        --target pdk4 \
                        --accelerator gpu  --devices 1 \
                        --opt_bound 1.0
```
