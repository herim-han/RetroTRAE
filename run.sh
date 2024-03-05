export CUDA_VISIBLE_DEVICES=0

nohup python ../train_gru.py  --sp_model ../../data/sp/ais_vocab_10.model \
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
                        --valid_filename ../../data/src/ais_vocab_10_valid.txt \
            			--train_filename ../../data/src/ais_vocab_10_train.txt \
			            1> cvae_ais_10.log 2>&1 &

#python ../train_gru.py  --sp_model ../../data/sp/ais_vocab_25.model \
#                        --train_ratio 0.00001 \
#                        --valid_ratio 0.00001 \
#                        --max_epochs 300  \
#                        --batch_size 1 \
#                        --accelerator gpu \
#                        --devices 1 \
#                        --check_val_every_n_epoch 5 \
#			             --strategy='ddp'\
#                        --lr  5e-5 \
#                        --masking_ratio 0 \
#                        --valid_filename ../../data/src/ais_vocab_25_valid.txt \
#			--train_filename ../../data/src/ais_vocab_25_train.txt 
