
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

from models import (BERTDecoder,
                    DINOv2Decoder,
                    dino_cls,
                    dino_mean,
                    bert_cls,
                    bert_pooled,
                    BertMeanPoolingAdapter,
                    BertPooledAdapter,
                    CLSAdapter,
                    MeanPoolingAdapter,
                    BERTEmbeddingDecoder,
                    DINOv2EmbeddingDecoder
)
from data import Multimodal3DIdent, Modality, DeMuReFusionConfig

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
parser.add_argument('--modality_names', nargs='+', default=['M1', 'M2'])


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
    
    print("load decoder")
    decoder_1, decoder_2 = DINOv2EmbeddingDecoder(), BERTEmbeddingDecoder()

    # Load data
    print("load data")
    data_dir = Path(args.data_dir)
    print(data_dir) 
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"
    trainset = Multimodal3DIdent(data_dir, mode="train", modality_names= (M1, M2))
    valset = Multimodal3DIdent(data_dir, mode="val", modality_names= (M1, M2))
    testset = Multimodal3DIdent(data_dir, mode="test", modality_names= (M1, M2))
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
    #valid_loader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
    #test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
    input_dims = get_dataloader_input_size(train_loader)
    # Train the dissentangler 
    m1 = Modality(
        name="image",
        encoder=encoder_1,
        decoder=decoder_1,
        zm_size=args.zm_size,
        input_shape=input_dims["image"],
        representation_fn=dino_cls
    )

    m2 = Modality(
        name="text",
        encoder=encoder_2,
        decoder=decoder_2,
        zm_size=args.zm_size,
        input_shape=input_dims["text"],
        representation_fn=bert_pooled
    )
    fusion_config = DeMuReFusionConfig(
    modalities = {
        m1.name.value: m1,
        m2.name.value: m2
    },
    shared_size=args.zs_size,
    num_train_samples=len(trainset),
    )
    rep_disentangler = DeMuReFusion(encoders=fusion_config.encoders, 
                                 decoders=fusion_config.decoders, 
                                 num_train_samples=fusion_config.num_train_samples,
                                 shared_size=fusion_config.shared_size,
                                 modality_names=fusion_config.modality_names, 
                                 z_sizes=fusion_config.z_sizes,
                                 modality_shapes=fusion_config.modality_shapes,
                                 mask=True,
                                 beta=args.beta,
                                 gamma=args.gamma,
                                 ckpt_path=args.ckpt_path,
                                 representation_fns=fusion_config.representation_fns,)  
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
