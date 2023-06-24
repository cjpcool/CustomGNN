python custom_gnn.py --cuda_device 2 --dataset ogbn-arxiv --para_name ogbn-arxiv --pw_adj_name ogbn-arxiv \
  --learning_rate 0.01 --seed 42 --schedule_patience 50 --patience 300 \
  --lam_pw_emd 1.0 --order 4 --path_length 6 --window_size 5 --lstm_hidden_units 64 --embedding_dim 64 \
  --batch_size 2000 --lam_pw_emd 1.0 --nhid 256 --T 0.2 \
  --use_triple 0 --lam_tri 0.01 --lam_tri_lstm 1.0 --margin 1 --samp_neg 500 --samp_pos 500 --K 2 \
  --dropout_pw 0.2 --dropout_enc 0.2 --dropout_adj 0.2 --dropout_input 0.2 --dropout_hidden 0.2