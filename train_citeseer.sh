python custom_gnn.py --cuda_device 0 --dataset citeseer --lam_pw_emd 10 --order 4 --path_length 10 --window_size 5 \
 --dropout_enc 0.6 --lstm_hidden_units 128 --batch_size 300 --seed 77 --schedule_patience 100 --learning_rate 0.01 \
 --use_triple 1 --lam_tri 0.1 --lam_tri_lstm 1.0 --margin 1.0