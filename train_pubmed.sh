python custom_gnn.py --cuda_device 3 --dataset pubmed --para_name pubmed --pw_adj_name pubmed \
  --learning_rate 0.1 --seed 42 --schedule_patience 50 --patience 200\
  --lam_pw_emd 1.0 --order 5 --path_length 6 --window_size 5 --lstm_hidden_units 128 --embedding_dim 256\
  --batch_size 2000 --lam_pw_emd 1.0 --nhid 128 --T 0.2\
  --use_triple 1 --lam_tri 0.01 --lam_tri_lstm 1.0 --margin 1 --samp_neg 5000 --samp_pos 5000 --K 4\
  --dropout_pw 0.6 --dropout_enc 0.5 --dropout_adj 0.5 --dropout_input 0.5 --dropout_hidden 0.8