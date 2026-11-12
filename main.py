
import sys
import os
import numpy as np
import argparse
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# sys.path.append('/home/sana/ml4h')
# sys.path.append('/home/sana')
import tensorflow as tf
import json
import keras 
from pathlib import Path
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_theme

from fusion import DeMuReFusion
from utils import *

tf.keras.callbacks.TerminateOnNaN()
tf.config.list_physical_devices('GPU')

import keras_hub

from models import BERTDecoder, DINOv2Decoder
from data import Multimodal3DIdent

parser = argparse.ArgumentParser(description='DeMuRe Fusion')
parser.add_argument('--log_path', default='/home/apxye/multimodal/ID3/logs/out.txt')
parser.add_argument('--plot_path', default='/home/apxye/multimodal/ID3/plots')
parser.add_argument('--ckpt_path', default='/home/apxye/multimodal/ID3/checkpoints')
parser.add_argument('--data_dir', default="/Users/moritz/Downloads/m3di")
parser.add_argument('--zm_size', default=400, type=int)
parser.add_argument('--zs_size', default=600, type=int)
parser.add_argument('--beta', default=1e-6)
parser.add_argument('--gamma', default=1)
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--modality_names', nargs='+', default=['img', 'txt'])


def main():
    args = parser.parse_args()
    # orig_stdout = sys.stdout
    # out_file = open(args.log_path, 'w')
    # sys.stdout = out_file

    # with open('/home/sana/multimodal/config.json', 'r') as f:
    #     configs = json.load(f)

    # if not os.path.exists(args.ckpt_path):
    #     os.mkdir(args.ckpt_path)
    # if not os.path.exists(args.plot_path):
    #     os.mkdir(args.plot_path)

    M1 = args.modality_names[0]
    M2 = args.modality_names[1]
    #data_paths = {M1:configs[M1]["raw_data_path"], M2:configs[M2]["raw_data_path"]}
    
    # Load pretrain models.
    print("load encoder")
    encoder_1, encoder_2 = load_pretrained_hub_models(
        "dinov2_base",
        "bert_small_en_uncased"
    )
    # img_dim = encoder_1.output_shape[-1]
    # text_dim = encoder_2.output_shape[0][-1]
    # print(img_dim, text_dim)
    print("load decoder")
    decoder_1, decoder_2 = DINOv2Decoder(), BERTDecoder()

    # Load data
    print("load data")
    data_dir = Path(args.data_dir)
    print(data_dir) 
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"
    trainset = Multimodal3DIdent(data_dir, mode="train")
    valset = Multimodal3DIdent(data_dir, mode="val")
    testset = Multimodal3DIdent(data_dir, mode="test")
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)
    
    # Train the dissentangler 
    rep_disentangler = DeMuReFusion(encoders={M1: encoder_1, M2:encoder_2}, 
                                 decoders={M1: decoder_1, M2:decoder_2}, 
                                 num_train_samples=len(trainset), shared_size=args.zs_size, modality_names=args.modality_names, 
                                 z_sizes={M1:args.zm_size, M2:args.zm_size},
                                 modality_shapes={M1: (img_dim, ), M2: (text_dim, )},
                                 mask=True, beta=args.beta, gamma=args.gamma, ckpt_path=args.ckpt_path)  
    del decoder_1, encoder_1, encoder_2, decoder_2


    dec_loss, enc_loss, shared_loss, modality_loss = rep_disentangler.train(train_loader, epochs_enc=args.n_epochs, epochs_dec=args.n_epochs, 
                                                                            lr_dec=1e-3, lr_enc=1e-3, iteration_count=20,
                                                                            extra_encoder_training=20, no_mask_epochs=2
                                                                            )
    
    _, axs = plt.subplots(1,3, figsize=(12, 4))
    axs[0].plot(modality_loss)
    axs[0].set_title("modality-specific training loss")
    axs[1].plot(shared_loss)
    axs[1].set_title("shared training loss")
    axs[2].plot(enc_loss)
    axs[2].set_title("Encoder training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_path,"train_curves.pdf"))


if __name__=="__main__": 
    main()
