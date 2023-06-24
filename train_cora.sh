python custom_gnn.py --cuda_device 3 --dataset cora --para_name cora --pw_adj_name cora \
  --learning_rate 0.01 --seed 42 --schedule_patience 50 --patience 300 --embedding_dim 512\
  --lam_pw_emd 1.0 --order 8 --path_length 12 --window_size 8 --lstm_hidden_units 128 --batch_size 200 \
  --use_triple 1 --lam_tri 0.01 --lam_tri_lstm 1.0 --margin 0.1 --samp_neg 5000 --samp_pos 15000 --K 4\
  --dropout_pw 0.6 --dropout_enc 0.5 --dropout_adj 0.5 --dropout_input 0.5 --dropout_hidden 0.5