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

data_dir = Path("/home/apxye/multimodal/ID3/m3di")
zm_size = 400
zs_size = 600
M1 = "image"
M2 = "text"

# Load pretrain models.
print("load encoder")
encoder_1, encoder_2 = load_pretrained_hub_models(
    "dinov2_base",
    "bert_small_en_uncased"
)

print("load decoder")
decoder_1, decoder_2 = DINOv2EmbeddingDecoder(), BERTEmbeddingDecoder()


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
    zm_size=zm_size,
    input_shape=input_dims["image"],
    representation_fn=dino_cls
)

m2 = Modality(
    name="text",
    encoder=encoder_2,
    decoder=decoder_2,
    zm_size=zm_size,
    input_shape=input_dims["text"],
    representation_fn=bert_pooled
)
fusion_config = DeMuReFusionConfig(
modalities = {
    m1.name.value: m1,
    m2.name.value: m2
},
shared_size=zs_size,
num_train_samples=len(trainset),
)
experiments = [
    Path("/home/apxye/multimodal/ID3/exp/1epoch5iter/mi"),
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
    



rep_disentangler = DeMuReFusionMI(encoders=fusion_config.encoders, 
                                    decoders=fusion_config.decoders, 
                                    num_train_samples=fusion_config.num_train_samples,
                                    shared_size=fusion_config.shared_size,
                                    modality_names=fusion_config.modality_names, 
                                    z_sizes=fusion_config.z_sizes,
                                    modality_shapes=fusion_config.modality_shapes,
                                    mask=True,
                                    beta=1e-6,
                                    gamma=1,
                                    ckpt_path=experiments[0] / "checkpoints",
                                    representation_fns=fusion_config.representation_fns,
                                    )  
del decoder_1, encoder_1, encoder_2, decoder_2

rep_disentangler.load_from_tf_checkpoint(4)
# dict with modality specific representations z_m, z_s per sample 
# z_s = outputs[modality][idx][shared] as numpy array 


from torch.utils.data import DataLoader, Subset

def make_small_loader(original_loader, n_samples: int = 100):
    orig_ds = original_loader.dataset
    small_ds = Subset(orig_ds, range(min(n_samples, len(orig_ds))))
    # Labels-Referenz für _extract_single_split durchreichen:
    small_ds.labels = orig_ds.labels
    return DataLoader(small_ds, batch_size=original_loader.batch_size, shuffle=False, collate_fn=collate_multimodal_fn)

# Anwendung:
# small_train_loader = make_small_loader(valid_loader, n_samples=30)
# small_test_loader  = make_small_loader(test_loader, n_samples=10)

# train_outputs = rep_disentangler.get_encoder_outputs(small_train_loader)
# test_outputs  = rep_disentangler.get_encoder_outputs(small_test_loader)

train_outputs = rep_disentangler.get_encoder_outputs(train_loader)
test_outputs = rep_disentangler.get_encoder_outputs(test_loader)

df_results = run_eval_to_csv_and_plot(
    train_outputs=train_outputs,
    test_outputs=test_outputs,
    train_loader=train_loader,
    test_loader=test_loader,
    factors=["object_shape", "text_phrasing", "spotlight_pos"],
    csv_path=str(experiments[0] / "plots" / "disentanglement_results_4.csv"),
    plot_path=str(experiments[0] / "plots" / "disentanglement_plot_4.png")
)

print(df_results.to_string(index=False))

