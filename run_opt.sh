#python bk_optimize.py      --sp_model ../data/sp/ais_vocab_25.model \
#                        --ckpt_path ckpt_file/run_ais_25/epoch=29.ckpt \
#                        --csv_path testtest \
#                        --n_repeat 1 --num_ask 10 --tokenize_method ais \
#                        --strategy='ddp' --opt_iter 2 \
#                        --opt_method botorch --target pdk4 \
#                        --accelerator gpu  --devices 1 \
#                        --opt_bound 1.0


python optimize.py      --sp_model ../data/sp/ais_vocab_25.model \
                        --ckpt_path cvae_ckpt_file/run_ais_25/epoch=29.ckpt \
                        --csv_path testtest \
                        --n_repeat 1 --num_ask 5 --tokenize_method ais \
                        --strategy='ddp' --opt_iter 2 \
                        --target pdk4 \
                        --accelerator gpu  --devices 1 \
                        --opt_bound 1.0
