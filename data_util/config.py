import os

import os
import torch
from numpy import random

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
#train_data_path = "../../dzzw-data/data/finished_files/chunked/train_*"
train_data_path =  "../../textsum/e-data2/finished_files/chunked/train_*"
#eval_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/val.bin")
#eval_data_path = "../../textsum/e-data-testval/finished_files/val.bin"
decode_data_path = "../../textsum/e-data_test2/finished_files/chunked/val_*"
#decode_data_path = "../../textsum/e-data-testval/finished_files/val.bin"
#decode_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/test.bin")
#vocab_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/vocab")
#vocab_path = "../../dzzw-data/data/finished_files/vocab"
vocab_path = "../../textsum/e-data2/finished_files/vocab"
save_model_path = '/root/my_pointer_summarizer/log/train_1610702047/model/'
log_root = "../log"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=2000
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

lr_coverage=0.15
