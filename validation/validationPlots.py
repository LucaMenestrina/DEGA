import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# data
DESeq2noShrinkage = pd.read_csv(
    "validation/DESeq2_bottomlyResults.csv", index_col=0)[["log2FoldChange", "padj"]].rename(columns={"log2FoldChange": "LFC", "padj": "FDR"})
DEGAnoShrinkage = pd.read_csv("validation/DEGA_bottomlyResults.csv",
                              index_col=0)[["log2 fold change (MLE) (strain)", "adjusted p-value (strain)"]].rename(columns={"log2 fold change (MLE) (strain)": "LFC", "adjusted p-value (strain)": "FDR"})
DESeq2withShrinkage = pd.read_csv(
    "validation/DESeq2_bottomlyWithShrinkageResults.csv", index_col=0)[["log2FoldChange", "padj"]].rename(columns={"log2FoldChange": "LFC", "padj": "FDR"})
DEGAwithShrinkage = pd.read_csv("validation/DEGA_bottomlyWithShrinkageResults.csv",
                                index_col=0)[["log2 fold change (MAP) (strain)", "adjusted p-value (strain)"]].rename(columns={"log2 fold change (MAP) (strain)": "LFC", "adjusted p-value (strain)": "FDR"})


def compare(subject, path=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 8), constrained_layout=True)
    fig.suptitle(r"$log_{2}$ Fold Change" if subject
                 == "LFC" else "Adjusted p-value")
    y = DEGAwithShrinkage[subject]
    x = DESeq2withShrinkage[subject]
    notNaN = list(set(np.where(~np.isnan(x))[0]).intersection(
        np.where(~np.isnan(y))[0]))
    y = y[notNaN]
    x = x[notNaN]
    sns.regplot(x=x, y=y, ci=95, n_boot=1000, scatter_kws={
                "s": 1.5, "alpha": 0.6}, line_kws={"linewidth": 1, "alpha": 0.6}, ax=ax1)
    rmse = mean_squared_error(x, y, squared=False)
    r2 = r2_score(x, y)
    ax1.text(max(x), min(y), f"With Shrinkage\nRMSE: {round(rmse,4)}\n"+r"$R^{2}$"
             + f": {round(r2, 4)}", horizontalalignment="right", verticalalignment="bottom")
    # ax1.set_title("With Shrinkage")
    ax1.set_ylabel("DESeq2")
    ax1.set_xlabel("DEGA")
    y = DEGAnoShrinkage[subject]
    x = DESeq2noShrinkage[subject]
    notNaN = list(set(np.where(~np.isnan(x))[0]).intersection(
        np.where(~np.isnan(y))[0]))
    y = y[notNaN]
    x = x[notNaN]
    sns.regplot(x=x, y=y, ci=95, n_boot=1000, scatter_kws={
                "s": 1.5, "alpha": 0.6}, line_kws={"linewidth": 1, "alpha": 0.6}, ax=ax2)
    rmse = mean_squared_error(x, y, squared=False)
    r2 = r2_score(x, y)
    ax2.text(max(x), min(y),
             f"Without Shrinkage\nRMSE: {round(rmse,4)}\n"+r"$R^{2}$"+f": {round(r2, 4)}", horizontalalignment="right", verticalalignment="bottom")
    # ax2.set_title("Without Shrinkage")
    ax2.set_ylabel("DESeq2")
    ax2.set_xlabel("DEGA")
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


compare("LFC", path="validation/LFC.svg")
compare("FDR", path="validation/FDR.svg")
