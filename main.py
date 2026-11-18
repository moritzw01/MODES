import nltk.redos
nltk.redos.DEFAULT_TIMEOUT = None
import sys
import os
import numpy as np
import argparse
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import json
import keras 
from pathlib import Path
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_theme

from fusion import DeMuReFusion, DeMuReFusionMI
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
from evaluation import run_eval_to_csv_and_plot





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
parser.add_argument('--iter_count', default=10, type=int)
parser.add_argument('--modality_names', nargs='+', default=['M1', 'M2'])
parser.add_argument('--model', default='default')


def main():
    args = parser.parse_args()

    M1 = args.modality_names[0]
    M2 = args.modality_names[1]
    
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
    trainset = Multimodal3DIdent(data_dir, mode="train", modality_names= (M1, M2))
    valset = Multimodal3DIdent(data_dir, mode="val", modality_names= (M1, M2))
    testset = Multimodal3DIdent(data_dir, mode="test", modality_names= (M1, M2))
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
    valid_loader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
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
    experiments = [
        Path("/home/apxye/multimodal/ID3/exp/1epoch5iter/l2=0"),
        Path("/home/apxye/multimodal/ID3/exp/1epoch5iter/modes")
    ] 
    for path in experiments:

        path.mkdir(parents=True, exist_ok=True)
        chck = path / "checkpoints"
        plots = path / "plots"
        logs = path / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        chck.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)
        


    model_name = args.model
    if model_name == "default":
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
                                         ckpt_path=experiments[1] / "checkpoints",
                                         representation_fns=fusion_config.representation_fns,
                                         )  
        del decoder_1, encoder_1, encoder_2, decoder_2
    
        dec_loss, enc_loss, shared_loss, modality_loss = rep_disentangler.train(train_loader, epochs_enc=args.n_epochs, epochs_dec=args.n_epochs, 
                                                                                lr_dec=1e-3, lr_enc=1e-3, iteration_count=args.iter_count,
                                                                                extra_encoder_training=3, no_mask_epochs=2
                                                                                )
        
        _, axs = plt.subplots(1,3, figsize=(12, 4))
        axs[0].plot(modality_loss)
        axs[0].set_title("modality-specific training loss")
        axs[1].plot(shared_loss)
        axs[1].set_title("shared training loss")
        axs[2].plot(enc_loss)
        axs[2].set_title("Encoder training loss")
        plt.tight_layout()
        plt.savefig(experiments[1] / "plots" / "train_curves.pdf")
    
    
        rep_disentangler.load_from_checkpoint()
        # dict with modality specific representations z_m, z_s per sample 
        # z_s = outputs[modality][idx][shared] as numpy array 
        train_outputs = rep_disentangler.get_encoder_outputs(train_loader)
        #valid_outputs = rep_disentangler.get_encoder_outputs(valid_loader)
        test_outputs = rep_disentangler.get_encoder_outputs(test_loader)
    
        df_results = run_eval_to_csv_and_plot(
            train_outputs=train_outputs,
            test_outputs=test_outputs,
            train_loader=train_loader,
            test_loader=test_loader,
            factors=["object_shape", "text_phrasing", "spotlight_pos"],
            csv_path=str(experiments[1] / "plots" / "disentanglement_results.csv"),
            plot_path=str(experiments[1] / "plots" / "disentanglement_plot.png")
        )
    
        print(df_results.to_string(index=False))
    
    elif model_name == 'mutual_info': 
        mi_rep_disentangler = DeMuReFusionMI(encoders=fusion_config.encoders, 
                                        decoders=fusion_config.decoders, 
                                        num_train_samples=fusion_config.num_train_samples,
                                        shared_size=fusion_config.shared_size,
                                        modality_names=fusion_config.modality_names, 
                                        z_sizes=fusion_config.z_sizes,
                                        modality_shapes=fusion_config.modality_shapes,
                                        mask=True,
                                        beta=args.beta,
                                        gamma=args.gamma,
                                        ckpt_path=experiments[0] / "checkpoints",
                                        representation_fns=fusion_config.representation_fns,
                                        l2=0)

        dec_loss, enc_loss, shared_loss, modality_loss = mi_rep_disentangler.train(train_loader, epochs_enc=args.n_epochs, epochs_dec=args.n_epochs, 
                                                                                    lr_dec=1e-3, lr_enc=1e-3, iteration_count=args.iter_count,
                                                                                    extra_encoder_training=3, no_mask_epochs=2
                                                                                )
        _, axs = plt.subplots(1,3, figsize=(12, 4))
        axs[0].plot(modality_loss)
        axs[0].set_title("modality-specific training loss")
        axs[1].plot(shared_loss)
        axs[1].set_title("shared training loss")
        axs[2].plot(enc_loss)
        axs[2].set_title("Encoder training loss")
        plt.tight_layout()
        plt.savefig(experiments[0] / "plots" / "train_curves.pdf")

        mi_rep_disentangler.load_from_checkpoint()
        # dict with modality specific representations z_m, z_s per sample 
        # z_s = outputs[modality][idx][shared] as numpy array 
        train_outputs = mi_rep_disentangler.get_encoder_outputs(train_loader)
        #valid_outputs = rep_disentangler.get_encoder_outputs(valid_loader)
        test_outputs = mi_rep_disentangler.get_encoder_outputs(test_loader)

        df_results = run_eval_to_csv_and_plot(
            train_outputs=train_outputs,
            test_outputs=test_outputs,
            train_loader=train_loader,
            test_loader=test_loader,
            factors=["object_shape", "text_phrasing", "spotlight_pos"],
            csv_path=str(experiments[0] / "plots" / "disentanglement_results.csv"),
            plot_path=str(experiments[0] / "plots" / "disentanglement_plot.png")
        )

        print(df_results.to_string(index=False))



    


if __name__=="__main__": 
    main()
