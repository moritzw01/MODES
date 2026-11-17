from sklearn.linear_model import SGDClassifier, SGDRegressor
from data import Multimodal3DIdent

from torch.utils.data import DataLoader
from pathlib import Path


from utils import collate_multimodal_fn
from fusion import DeMuReFusion
import numpy as np 
# data_dir = Path("")
# M1, M2 = "image", "text"

# trainset = Multimodal3DIdent(data_dir, mode="train", modality_names= (M1, M2))
# valset = Multimodal3DIdent(data_dir, mode="val", modality_names= (M1, M2))
# testset = Multimodal3DIdent(data_dir, mode="test", modality_names= (M1, M2))
# train_loader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
# valid_loader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)
# test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_multimodal_fn)

# model = DeMuReFusion()


# # dict with modality specific representations z_m, z_s per sample 
# # z_s = outputs[modality][idx][shared] as numpy array 
# train_outputs = model.get_encoder_outputs(train_loader)
# valid_outputs = model.get_encoder_outputs(valid_loader)
# test_outputs = model.get_encoder_outputs(test_loader)

factors = ["object_shape", "text_phrasing", "spotlight_position"]
# object shape is a shared variable classification, tetx phrasing text specific classification 
# and spotlight position image specific regression task 
# def train_and_predict(outputs: dict, outputs_test: dict, dataloader:DataLoader): 
#     prediction_results = {
#         factor: {
#             mod: {
#                 "shared": None,
#                 "specific": None,
#                 "both": None
#             }
#             for mod,_ in outputs.items()
#         }
#         for factor in factors
#     }

#     for mod, id_dict in outputs.items(): 
#         for factor in factors: 
#             if factor == "object_shape" or factor == "text_phrasing": 
#                 #classification 
#                 experiments = []
#                 for id, z_dict in id_dict.items(): 
#                     z_s, z_m = z_dict["shared"], z_dict["specific"]
#                     sample = dataloader[id]
                    
#                     x = [z_m, z_s, np.concat((z_m, z_s))]
#                     experiments.append(x)



#                     gt = sample[mod].z[factor]




import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. ADAPTER (Train vs. Test Split)
# ==========================================
import numpy as np
import pandas as pd


def _extract_single_split(outputs: dict, dataloader, factors: tuple):
    """
    Liest die Latent-Repräsentationen aus `outputs` und die Ground-Truth-Faktoren
    direkt aus `dataloader.dataset.labels` (ohne Bilder von der Festplatte zu laden!).
    """
    dataset = getattr(dataloader, "dataset", dataloader)
    modalities = list(outputs.keys())
    sample_ids = sorted(outputs[modalities].keys())

    z_dict = {mod: {"shared": [], "specific": [], "both": []} for mod in modalities}

    for sample_id in sample_ids:
        for mod in modalities:
            z_s = np.asarray(outputs[mod][sample_id]["shared"]).reshape(-1)
            z_m = np.asarray(outputs[mod][sample_id]["specific"]).reshape(-1)

            z_dict[mod]["shared"].append(z_s)
            z_dict[mod]["specific"].append(z_m)
            z_dict[mod]["both"].append(np.concatenate([z_m, z_s], axis=-1))

    for mod in modalities:
        for rep_type in ["shared", "specific", "both"]:
            z_dict[mod][rep_type] = np.stack(z_dict[mod][rep_type], axis=0)

    # Direkter Zugriff auf die DataFrames dataset.labels["z_image"] und dataset.labels["z_text"]
    z_image_df = dataset.labels["z_image"]
    z_text_df = dataset.labels["z_text"]

    y_cols = {}
    for factor in factors:
        if factor in z_image_df.columns:
            y_cols[factor] = z_image_df[factor].iloc[sample_ids].to_numpy()
        elif factor in z_text_df.columns:
            y_cols[factor] = z_text_df[factor].iloc[sample_ids].to_numpy()
        else:
            raise KeyError(
                f"Faktor '{factor}' weder in z_image ({list(z_image_df.columns)}) "
                f"noch in z_text ({list(z_text_df.columns)}) gefunden!"
            )

    return z_dict, pd.DataFrame(y_cols)




def adapt_train_test_for_paper_eval(
    train_outputs: dict,
    test_outputs: dict,
    train_loader,
    test_loader,
    factors=("object_shape", "text_phrasing", "spotlight_position")
):
    z_tr, y_tr = _extract_single_split(train_outputs, train_loader, factors)
    z_te, y_te = _extract_single_split(test_outputs, test_loader, factors)

    n_train = len(y_tr)
    n_test = len(y_te)

    z_combined = {}
    for mod in z_tr.keys():
        z_combined[mod] = {
            rep_type: np.concatenate([z_tr[mod][rep_type], z_te[mod][rep_type]], axis=0)
            for rep_type in ["shared", "specific", "both"]
        }

    y_combined = pd.concat([y_tr, y_te], axis=0, ignore_index=True)
    single_split_indices = [(np.arange(0, n_train), np.arange(n_train, n_train + n_test))]

    return z_combined, y_combined, single_split_indices


def phenotype_predictor(
    z_train, y_train, z_test, y_test, phenotypes, kfold_indices,
    regression_phenos=("spotlight_position",), mask=None, n_pca=None
):
    phenotypes_scores = {}
    if mask is not None:
        if tf.reduce_sum(mask) == 0:
            for pheno in phenotypes:
                phenotypes_scores[pheno] = (0, 0)
            return phenotypes_scores
        z_test = np.take(z_test, np.argwhere(mask == 1)[:, 0], axis=1)
        z_train = np.take(z_train, np.argwhere(mask == 1)[:, 0], axis=1)

    for pheno in phenotypes:
        if n_pca == 0 or z_train.shape[-1] == 0:
            phenotypes_scores[pheno] = [[0], [0]]
            continue

        n_classes = len(np.unique(y_train[pheno].to_numpy()))
        is_regression = pheno in regression_phenos

        # --- Zweig A: Regression (spotlight_position) ---
        if is_regression:
            r2_test, r2_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor = make_pipeline(StandardScaler(with_mean=True), Ridge(solver='lsqr', max_iter=250000))
                X_train, X_test = z_train[train_index], z_train[test_index]
                Y_train, Y_test = y_train[pheno].to_numpy()[train_index], y_train[pheno].to_numpy()[test_index]
                if n_pca is not None and n_pca < X_train.shape[-1]:
                    pca = PCA(n_components=n_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                predictor.fit(X_train, Y_train)
                r2_test.append(r2_score(Y_test, predictor.predict(X_test)))
                r2_train.append(r2_score(Y_train, predictor.predict(X_train)))
            phenotypes_scores[pheno] = [r2_train, r2_test]
        else:
            acc_test, acc_train = [], []
            for (train_index, test_index) in kfold_indices:
                # Multinomiale LogisticRegression erzeugt saubere Softmax-Klassengebiete
                predictor = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', l1_ratio=0.5, max_iter=1000)
                X_train, X_test = z_train[train_index], z_train[test_index]
                Y_train, Y_test = y_train[pheno].to_numpy()[train_index], y_train[pheno].to_numpy()[test_index]
                if n_pca is not None and n_pca < X_train.shape[-1]:
                    pca = PCA(n_components=n_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                predictor.fit(X_train, Y_train)
                acc_test.append(accuracy_score(Y_test, predictor.predict(X_test)))
                acc_train.append(accuracy_score(Y_train, predictor.predict(X_train)))
            phenotypes_scores[pheno] = [acc_train, acc_test]

    return phenotypes_scores


def phenotype_predictor_lf(
    z_train_1, y_train_1, z_train_2, y_train_2, phenotypes, kfold_indices,
    regression_phenos=("spotlight_position",)
):
    phenotypes_scores = {}
    for pheno in phenotypes:
        n_classes = len(np.unique(y_train_1[pheno].to_numpy()))
        is_regression = pheno in regression_phenos

        # --- Zweig A: Regression (spotlight_position) ---
        if is_regression:
            r2_test, r2_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor_1 = SGDRegressor()
                predictor_2 = SGDRegressor()
                predictor_1.fit(z_train_1[train_index], y_train_1[pheno].to_numpy()[train_index])
                predictor_2.fit(z_train_2[train_index], y_train_2[pheno].to_numpy()[train_index])

                z_pred_train = (predictor_1.predict(z_train_1[train_index]) + predictor_2.predict(z_train_2[train_index])) / 2
                z_pred = (predictor_1.predict(z_train_1[test_index]) + predictor_2.predict(z_train_2[test_index])) / 2

                r2_test.append(r2_score(y_train_1[pheno].to_numpy()[test_index], z_pred))
                r2_train.append(r2_score(y_train_1[pheno].to_numpy()[train_index], z_pred_train))
            phenotypes_scores[pheno] = [r2_train, r2_test]

        # --- Zweig B: Binäre Klassifikation ---
        elif n_classes == 2:
            auc_test, auc_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor_1 = SGDClassifier()
                predictor_2 = SGDClassifier()
                predictor_1.fit(z_train_1[train_index], y_train_1[pheno].to_numpy()[train_index])
                predictor_2.fit(z_train_2[train_index], y_train_2[pheno].to_numpy()[train_index])

                z_pred_train = (predictor_1.predict(z_train_1[train_index]) + predictor_2.predict(z_train_2[train_index])) / 2
                z_pred = (predictor_1.predict(z_train_1[test_index]) + predictor_2.predict(z_train_2[test_index])) / 2

                auc_test.append(roc_auc_score(y_train_1[pheno].to_numpy()[test_index], z_pred))
                auc_train.append(roc_auc_score(y_train_1[pheno].to_numpy()[train_index], z_pred_train))
            phenotypes_scores[pheno] = [auc_train, auc_test]

        # --- Zweig C: Multiclass Late Fusion (log_loss -> Softmax-Wahrscheinlichkeiten mitteln) ---
        else:
            acc_test, acc_train = [], []
            for (train_index, test_index) in kfold_indices:
                predictor_1 = SGDClassifier(loss="log_loss")
                predictor_2 = SGDClassifier(loss="log_loss")
                predictor_1.fit(z_train_1[train_index], y_train_1[pheno].to_numpy()[train_index])
                predictor_2.fit(z_train_2[train_index], y_train_2[pheno].to_numpy()[train_index])

                probs_train = (predictor_1.predict_proba(z_train_1[train_index]) + predictor_2.predict_proba(z_train_2[train_index])) / 2

                
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_eval_to_csv_and_plot(
    train_outputs: dict,
    test_outputs: dict,
    train_loader,
    test_loader,
    factors=("object_shape", "text_phrasing", "spotlight_position"),
    regression_phenos=("spotlight_position",),
    csv_path="disentanglement_results.csv",
    plot_path="disentanglement_plot.png"
):
    """
    Führt die komplette Evaluation durch, speichert alle Metriken als CSV
    und erstellt einen übersichtlichen Bar-Chart-Vergleich als PNG/PDF.
    """
    z_all, y_all, split_indices = adapt_train_test_for_paper_eval(
        train_outputs=train_outputs,
        test_outputs=test_outputs,
        train_loader=train_loader,
        test_loader=test_loader,
        factors=factors
    )

    modalities = list(train_outputs.keys())
    mod1, mod2 = modalities, modalities
    rep_types = ["shared", "specific", "both"]

    rows = []

    for factor in factors:
        n_classes = len(np.unique(y_all[factor].to_numpy()))
        if factor in regression_phenos:
            metric_name = "R²"
        elif n_classes == 2:
            metric_name = "ROC-AUC"
        else:
            metric_name = "Accuracy"

        # 1. Unimodale Auswertung (jede Modalität einzeln)
        for mod in modalities:
            for rep_type in rep_types:
                res = phenotype_predictor(
                    z_train=z_all[mod][rep_type],
                    y_train=y_all,
                    z_test=z_all[mod][rep_type],
                    y_test=y_all,
                    phenotypes=[factor],
                    kfold_indices=split_indices,
                    regression_phenos=regression_phenos
                )
                rows.append({
                    "Factor": factor,
                    "Metric": metric_name,
                    "Source": mod,
                    "Representation": rep_type,
                    "Train_Score": float(res[factor]),
                    "Test_Score": float(res[factor]),
                })

        # 2. Late Fusion Auswertung (beide Modalitäten gemittelt)
        if len(modalities) == 2:
            for rep_type in rep_types:
                res_lf = phenotype_predictor_lf(
                    z_train_1=z_all[mod1][rep_type],
                    y_train_1=y_all,
                    z_train_2=z_all[mod2][rep_type],
                    y_train_2=y_all,
                    phenotypes=[factor],
                    kfold_indices=split_indices,
                    regression_phenos=regression_phenos
                )
                rows.append({
                    "Factor": factor,
                    "Metric": metric_name,
                    "Source": "late_fusion",
                    "Representation": rep_type,
                    "Train_Score": float(res_lf[factor]),
                    "Test_Score": float(res_lf[factor]),
                })

    # DataFrame erstellen und als CSV speichern
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Ergebnisse erfolgreich gespeichert unter: {csv_path}")

    # ==========================================
    # MATPLOTLIB PLOT ERSTELLEN
    # ==========================================
    n_factors = len(factors)
    fig, axes = plt.subplots(1, n_factors, figsize=(5.5 * n_factors, 4.8), dpi=200)
    if n_factors == 1:
        axes = [axes]

    sources = modalities + (["late_fusion"] if len(modalities) == 2 else [])
    colors = {
        "shared": "#4C72B0",     # Blau
        "specific": "#DD8452",   # Orange
        "both": "#55A868",       # Grün
    }

    x = np.arange(len(sources))
    width = 0.25

    for ax, factor in zip(axes, factors):
        sub_df = df[df["Factor"] == factor]
        metric_label = sub_df["Metric"].iloc

        for i, rep_type in enumerate(rep_types):
            rep_scores = []
            for src in sources:
                val = sub_df[(sub_df["Source"] == src) & (sub_df["Representation"] == rep_type)]["Test_Score"].values
                rep_scores.append(val if len(val) > 0 else 0.0)

            offset = (i - 1) * width
            bars = ax.bar(x + offset, rep_scores, width, label=rep_type.capitalize(), color=colors[rep_type], edgecolor="black", linewidth=0.6)

            # Werte direkt über die Balken schreiben
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, max(h, 0)),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        ax.set_title(factor.replace("_", " ").title(), fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(f"Test {metric_label}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in sources], fontsize=10)
        ax.set_ylim(min(0.0, sub_df["Test_Score"].min() - 0.1), 1.12)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Gemeinsame Legende oben mittig
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Plot erfolgreich gespeichert unter: {plot_path}")

    return df


        