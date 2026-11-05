"""Data preparation, validation, and figures for Homework 6."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_data(root):
    folder = Path(root) / "data/aps_failure"
    frames = [
        pd.read_csv(
            folder / f"aps_failure_{s}_set.csv", skiprows=20, na_values="na"
        )
        for s in ["training", "test"]
    ]
    Xtr, Xte = [d.drop(columns="class") for d in frames]
    ytr, yte = [
        d["class"].map({"neg": 0, "pos": 1}).to_numpy() for d in frames
    ]
    assert Xtr.shape == (60000, 170)
    assert Xte.shape == (16000, 170)
    assert Xtr.columns.equals(Xte.columns)
    assert all(np.isfinite(y).all() for y in [ytr, yte])
    return Xtr, Xte, ytr, yte


def metrics(y, probability):
    pred = (probability > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        error=1 - accuracy_score(y, pred),
        auc=roc_auc_score(y, probability),
        recall=recall_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        TN=int(tn),
        FP=int(fp),
        FN=int(fn),
        TP=int(tp),
    )


def report_model(name, ytr, yte, ptr, pte, rows, predictions):
    for split, y, p in [("train", ytr, ptr), ("test", yte, pte)]:
        rows.append(dict(model=name, split=split, **metrics(y, p)))
        predictions[split][name] = p
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
    for ax, split, y, p in zip(
        axes[:2], ["Train", "Test"], [ytr, yte], [ptr, pte]
    ):
        cm = confusion_matrix(y, p > 0.5, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > cm.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["neg", "pos"],
            yticklabels=["neg", "pos"],
            xlabel="Predicted",
            ylabel="Actual",
            title=split,
        )
    for split, y, p in [("Train", ytr, ptr), ("Test", yte, pte)]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[2].plot(fpr, tpr, label=f"{split} AUC={roc_auc_score(y, p):.4f}")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[2].set(
        xlabel="False-positive rate", ylabel="True-positive rate", title="ROC"
    )
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle(name)
    fig.tight_layout()
    plt.show()


def show_exploration(X, y, features):
    rng = np.random.default_rng(42)
    sample = np.r_[
        rng.choice(np.flatnonzero(y == 0), 2000, replace=False),
        np.flatnonzero(y == 1),
    ]
    for kind in ["scatter", "box"]:
        fig, axes = plt.subplots(4, 4, figsize=(10, 8))
        for ax, feature in zip(axes.flat, features):
            values = np.log1p(X[feature].to_numpy())
            if kind == "scatter":
                jitter = rng.uniform(-0.06, 0.06, len(sample))
                ax.scatter(
                    y[sample] + jitter,
                    values[sample],
                    c=y[sample],
                    cmap="coolwarm",
                    s=3,
                    alpha=0.25,
                )
            else:
                ax.boxplot(
                    [values[y == k] for k in [0, 1]],
                    positions=[0, 1],
                    showfliers=False,
                    widths=0.5,
                )
            ax.set(
                xticks=[0, 1],
                xticklabels=["neg", "pos"],
                ylabel="log(1+x)",
                title=feature,
            )
        for ax in list(axes.flat)[len(features) :]:
            ax.axis("off")
        fig.suptitle("Highest-CV features: " + kind + " plots")
        fig.tight_layout()
        plt.show()


def xgb_model(alpha):
    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        reg_alpha=alpha,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        max_bin=128,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )


def validate_boosting(X, y, alphas, resample, output):
    rows, sizes = [], []
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (train, valid) in enumerate(folds.split(X, y), 1):
        imputer = SimpleImputer(strategy="median")
        Xt = imputer.fit_transform(X.iloc[train])
        Xv = imputer.transform(X.iloc[valid])
        scaler = StandardScaler().fit(Xt)
        Xt = scaler.transform(Xt).astype(np.float32)
        Xv = scaler.transform(Xv).astype(np.float32)
        yt = y[train]
        if resample:
            Xt, yt = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xt, yt)
        sizes.append(
            dict(
                fold=fold,
                original_rows=len(train),
                fitted_rows=len(yt),
                positives=int(yt.sum()),
                negatives=int((yt == 0).sum()),
                validation_rows=len(valid),
                validation_positives=int(y[valid].sum()),
            )
        )
        for alpha in alphas:
            model = xgb_model(alpha).fit(Xt, yt)
            p = model.predict_proba(Xv)[:, 1]
            rows.append(dict(fold=fold, alpha=alpha, **metrics(y[valid], p)))
    table = pd.DataFrame(rows)
    summary = table.groupby("alpha").agg(
        cv_error=("error", "mean"),
        sd_error=("error", "std"),
        cv_auc=("auc", "mean"),
        cv_recall=("recall", "mean"),
    )
    summary["se_error"] = summary.sd_error / np.sqrt(5)
    summary = summary.reset_index()
    best = summary.sort_values(["cv_error", "alpha"]).iloc[0]
    return table, summary, float(best.alpha), pd.DataFrame(sizes)
