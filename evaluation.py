from sklearn.linear_model import SGDClassifier, SGDRegressor
from data import Multimodal3DIdent

from torch.utils.data import DataLoader
from pathlib import Path


from utils import collate_multimodal_fn
from fusion import DeMuReFusion

data_dir = Path("")
M1, M2 = "image", "text"

trainset = Multimodal3DIdent(data_dir, mode="train", modality_names= (M1, M2))
valset = Multimodal3DIdent(data_dir, mode="val", modality_names= (M1, M2))
testset = Multimodal3DIdent(data_dir, mode="test", modality_names= (M1, M2))
train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
valid_loader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)

model = DeMuReFusion()


# dict with modality specific representations z_m, z_s per sample 
# z_s = outputs[modality][idx][shared] as numpy array 
train_outputs = model.get_encoder_outputs(train_loader)
valid_outputs = model.get_encoder_outputs(valid_loader)
test_outputs = model.get_encoder_outputs(test_loader)

factors = ["object_shape", "text_phrasing", "spotlight_position"]
# object shape is a shared variable classification, tetx phrasing text specific classification 
# and spotlight position image specific regression task 
def train_and_predict(outputs: dict, dataloader): 
    prediction_results = {
        factor: {
            mod: {
                "shared": None,
                "specific": None,
                "both": None
            }
            for mod,_ in outputs.items()
        }
        for factor in factors
    }

    for mod, id_dict in outputs.items(): 
        for factor in factors: 
            if factor == "object_shape" or factor == "text_phrasing": 
                #classification 
                for id, z_dict in id_dict.items(): 
                    pass

            





