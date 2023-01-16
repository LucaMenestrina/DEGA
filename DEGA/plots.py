import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import seaborn as sns
import networkx as nx

from sklearn.decomposition import PCA

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from DEGA.logger import log


def plotDispersions(dispersionsResults, baseMean, geneColor="black", fitColor="red",
                    finalColor="dodgerblue", s=2, linewidth=1, legend=True, path=None):
    plt.scatter(baseMean, dispersionsResults["dispGeneEst"],
                s=s, linewidth=0, c=geneColor)  # label="Gene-Wise Dispersion Estimate"
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(
        baseMean[~dispersionsResults["dispOutlier"]],
        dispersionsResults["dispersion"][~dispersionsResults["dispOutlier"]],
        s=s, linewidth=0, c=finalColor)  # label="Final Dispersion"
    plt.scatter(
        baseMean[dispersionsResults["dispOutlier"]],
        dispersionsResults["dispersion"][dispersionsResults["dispOutlier"]],
        s=7.5*s, linewidth=linewidth, facecolors="none", edgecolors=finalColor)
    plt.scatter(
        baseMean, dispersionsResults["dispFit"], s=s, linewidth=0, c=fitColor)  # label="Fitted"
    if legend:
        custom_elements = [
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color=geneColor,
                   label="Gene-Wise Dispersion Estimate"),
            Line2D([0], [0], linewidth=0, marker="o",
                   markersize=5, color=fitColor, label="Fitted"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5,
                   color=finalColor, label="Final Dispersion"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=8, markerfacecolor="none",
                   markeredgecolor=finalColor, label="Final Dispersion Outliers")
        ]
        plt.legend(handles=custom_elements)
    plt.xlabel("Mean of Normalized Counts")
    plt.ylabel("Dispersion")
    plt.tight_layout()
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotMA(testResults, baseMean, pValueThreshold=0.05, lfcThreshold=0, boundedY=True,
           legend=True, highlightGenes=None, labels=True, path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    if boundedY:
        yQuantile = np.quantile(testResults[LFCColumnName].abs(), 0.999)
        ylim = [-yQuantile*lfcThreshold, yQuantile*lfcThreshold] if lfcThreshold != 0 else [np.min(
            testResults[LFCColumnName]), np.max(testResults[LFCColumnName])]
    else:
        ylim = [np.min(testResults[LFCColumnName]),
                np.max(testResults[LFCColumnName])]
    markers = np.array(["^" if lfc > ylim[1] else "v" if lfc < ylim[0]
                       else "o" for lfc in testResults[LFCColumnName]])
    colors = np.array(["crimson" if (lfc > 0 and apv < pValueThreshold) else
                       "royalblue" if (lfc < 0 and apv < pValueThreshold) else
                       "dimgrey" for _, lfc, apv in testResults[[LFCColumnName, PvalueColumnName]].itertuples()])
    alpha = np.array(
        [0.75 if apv < pValueThreshold else 0.2 for apv in testResults[PvalueColumnName]])
    lfc = np.minimum(np.maximum(
        testResults[LFCColumnName], ylim[0]), ylim[1])
    plt.axhline(0, color="lightgrey", alpha=0.6, zorder=1)
    if lfcThreshold != 0:
        plt.axhline(lfcThreshold, color="lightgrey", alpha=0.8, zorder=100)
        plt.axhline(-lfcThreshold, color="lightgrey", alpha=0.8, zorder=100)
    for marker in ["^", "v", "o"]:
        where = np.where(markers == marker)[0]
        s = 5 if marker == "o" else 17.5
        if where.size:
            plt.scatter(baseMean[where], lfc[where],
                        s=s, linewidth=0, c=colors[where], marker=marker, alpha=alpha[where])
        if not highlightGenes is None:
            hgIdx = lfc[where].index.isin(highlightGenes)
            plt.scatter(baseMean[where][hgIdx], lfc[where][hgIdx], s=s*6,
                        marker="o", linewidth=2, facecolors="none", edgecolors="lime", alpha=0.8, zorder=10)
    if not highlightGenes is None and labels:
        hgIdx = lfc[where].index.isin(highlightGenes)
        for idx in range(len(highlightGenes)):
            plt.text(baseMean[where][hgIdx][idx]*1.02,
                     lfc[where][hgIdx][idx]*1.02, highlightGenes[idx], fontsize="xx-small", zorder=100)
    ymarg = (ylim[1]-ylim[0])/20
    plt.ylim((ylim[0]-ymarg, ylim[1]+ymarg))
    # plt.xlim([5, 1e5])
    plt.xscale("log")
    plt.xscale("log")
    if legend:
        custom_elements = [
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color="dimgrey",
                   label="Not Significant"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5,
                   color="crimson", label=f"Upregulated & Adjusted P-Value<{pValueThreshold}"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color="royalblue",
                   label=f"Downregulated & Adjusted P-Value<{pValueThreshold}"),
        ]
        if not highlightGenes is None:
            custom_elements += [Line2D([0], [0], linewidth=0, marker="o", markersize=8, markerfacecolor="none",
                                       markeredgecolor="lime", label="Highlighted Genes")]
        plt.legend(handles=custom_elements, fontsize="x-small")
    plt.ylabel(LFCColumnName)  # M
    plt.xlabel("Mean of Normalized Counts")  # A
    plt.suptitle("MA plot")
    up = (testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)][LFCColumnName] > lfcThreshold).sum()
    down = (testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)][LFCColumnName] < -lfcThreshold).sum()
    plt.title(
        f"Adjusted P-Value<{pValueThreshold}, |LFC|>{lfcThreshold}, UP: {up}, DOWN: {down}", fontsize="small")
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotVolcano(testResults, pValueThreshold=0.05, lfcThreshold=0, boundedY=True, labels=True,
                labelsQuantile=0.005, legend=True, highlightGenes=None, highlightedGenesLabels=True, path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    yThreshold = -np.log10(pValueThreshold)
    x = testResults[LFCColumnName]
    y = -np.log10(testResults[PvalueColumnName])
    yCeiling = y[~np.isnan(y)].quantile(0.999)
    ylim = [y[~np.isnan(y)].min(), yCeiling]
    markers = np.array(["^" if val > yCeiling else "o" for val in y])
    colors = np.array(["crimson" if (np.abs(X) > lfcThreshold and Y > yThreshold) else
                       "royalblue" if Y > yThreshold else
                       "forestgreen" if np.abs(X) > lfcThreshold else
                       "dimgrey" for X, Y in zip(x, y)])
    y = np.minimum(np.maximum(y, ylim[0]), ylim[1])
    for marker in ["^", "o"]:
        where = np.where(markers == marker)[0]
        s = 4 if marker == "o" else 15
        plt.scatter(x[where], y[where], s=s, linewidth=0,
                    c=colors[where], marker=marker, alpha=0.8)
    if highlightGenes:
        plt.scatter(x.loc[highlightGenes], y.loc[highlightGenes], s=25,  marker="o",
                    linewidth=2, facecolors="none", edgecolors="lime", zorder=10)
    plt.axhline(yThreshold, ls="--", c="black", linewidth=1, alpha=0.6)
    if lfcThreshold != 0:
        plt.axvline(-lfcThreshold, ls="--", c="black", linewidth=1, alpha=0.6)
        plt.axvline(lfcThreshold, ls="--", c="black", linewidth=1, alpha=0.6)
    significant = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)][[LFCColumnName, PvalueColumnName]]
    up = (significant[LFCColumnName] > lfcThreshold).sum()
    down = (significant[LFCColumnName] < -lfcThreshold).sum()
    if labels:
        if labelsQuantile and labelsQuantile != 1:
            # if a labelsQuantile is defined plot only labels
            # that are outside an ellipse with focuses on the p-value and LFC threshold
            nLabels = min(int(round(x.size*labelsQuantile, 0)),
                          significant.shape[0])
            def distance(coords): return (np.sqrt(np.sum(np.square(coords-np.array([lfcThreshold, yThreshold])))))+(
                np.sqrt(np.sum(np.square(coords-np.array([-lfcThreshold, yThreshold])))))
            coords = np.column_stack(
                (x.loc[significant.index].values, y.loc[significant.index].fillna(0).values))
            distances = np.apply_along_axis(distance, axis=1, arr=coords)
            for name, X, Y in np.column_stack((significant.index, coords))[distances.argpartition(-nLabels)[-nLabels:]]:
                if X > lfcThreshold:
                    plt.text(X*1.01, Y*1.01,
                             name, fontsize="xx-small", zorder=100)
                else:
                    plt.text(X*1.01, Y*1.01, name, fontsize="xx-small",
                             horizontalalignment="right", zorder=100)
        else:
            for name, X, Y in significant.itertuples():
                Y = np.minimum(np.maximum(-np.log10(Y), ylim[0]), ylim[1])
                if X > lfcThreshold:
                    plt.text(X*1.01, Y*1.01,
                             name, fontsize="xx-small", zorder=100)
                else:
                    plt.text(X*1.01, Y*1.01, name, fontsize="xx-small",
                             horizontalalignment="right", zorder=100)
    if highlightedGenesLabels:
        for name, X, Y in testResults[[LFCColumnName, PvalueColumnName]].loc[highlightGenes].itertuples():
            if X > lfcThreshold:
                plt.text(X*1.01, -np.log10(Y)*1.01,
                         name, fontsize="xx-small", zorder=100)
            else:
                plt.text(X*1.01, -np.log10(Y)*1.01, name, fontsize="xx-small",
                         horizontalalignment="right", zorder=100)
    if legend:
        custom_elements = [
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color="dimgrey",
                   label="Not Significant"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5,
                   color="royalblue", label=f"Adjusted P-Value<{pValueThreshold}"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color="forestgreen",
                   label=f"|LFC|>{lfcThreshold}"),
            Line2D([0], [0], linewidth=0, marker="o", markersize=5, color="crimson",
                   label=f"Adjusted P-Value<{pValueThreshold} & |LFC|>{lfcThreshold}"),
        ]
        if not highlightGenes is None:
            custom_elements += [Line2D([0], [0], linewidth=0, marker="o", markersize=8, markerfacecolor="none",
                                       markeredgecolor="lime", label="Highlighted Genes")]
        plt.legend(handles=custom_elements, fontsize="x-small")
    ymarg = (ylim[1]-ylim[0])/20
    plt.ylim(ylim[0]-ymarg, ylim[1]+ymarg)
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10 Adjusted P-Value")
    plt.suptitle("Volcano Plot of Differentially Expressed Genes")
    plt.title(
        f"Adjusted P-Value<{pValueThreshold}, LFC>{lfcThreshold}, UP: {up}, DOWN: {down}", fontsize="small")
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotSparsity(normalizedCounts, path=None):
    plt.scatter(normalizedCounts.sum(axis=1), normalizedCounts.max(
        axis=1)/normalizedCounts.sum(axis=1), s=5, c="cornflowerblue", alpha=0.6)
    plt.xscale("log")
    plt.ylabel("max count / sum")
    plt.xlabel("sum of counts per gene")
    plt.title("Concentration of counts over total sum of counts")
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotSamplesCorrelation(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                           pValueThreshold=0.05, lfcThreshold=0, method="spearman", clustering=True, path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)]
    if significantGenes.empty:
        log.error("There aren't any significant differentially expressed genes")
    else:
        significantCounts = transformedCounts.loc[significantGenes.index].copy(
        )
        significantCountsCorr = significantCounts.corr(method=method)
        labels = significantCountsCorr.index + \
            "\n(" + phenotypeData[phenotypeColumn] + ")"
        if clustering:
            plot = sns.clustermap(significantCountsCorr, cmap="coolwarm",
                                  cbar_pos=(1.01, 0.25, 0.05, 0.5), linewidth=0.1,
                                  annot=True, alpha=0.8, yticklabels=labels, xticklabels=labels)
            plot.ax_col_dendrogram.set_visible(False)
        else:
            sns.heatmap(significantCountsCorr, cmap="coolwarm", linewidth=0.1,
                        annot=True, alpha=0.8, yticklabels=labels, xticklabels=labels)
        plt.suptitle("Samples Correlation Plot", fontsize="x-large",
                     y=0.9 if clustering else None)
        if path is None:
            return plt.gca()
        else:
            plt.savefig(path, dpi=300)
            return plt.gca()


def plotSamplesGenesCorrelation(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                                pValueThreshold=0.05, lfcThreshold=0, metric="correlation", path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)]
    if significantGenes.empty:
        log.error("There aren't any significant differentially expressed genes")
    else:
        significantCounts = transformedCounts.loc[significantGenes.index].copy(
        )
        plot = sns.clustermap(significantCounts, metric="correlation",
                              cmap=mpl.cm.get_cmap("Pastel2", 3), cbar_kws={'ticks': [0.0, 0.5, 1.0]},
                              alpha=0.8, standard_scale=0,
                              xticklabels=(significantCounts.columns + "\n(" + phenotypeData[phenotypeColumn] + ")"))
        plot.ax_heatmap.set_ylabel("")
        plt.suptitle("Samples Genes Correlation Plot", fontsize="x-large")
        if path is None:
            return plt.gca()
        else:
            plt.savefig(path, dpi=300)
            return plt.gca()


def plotPCAExplainedVariance(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                             pValueThreshold=0.05, lfcThreshold=0, path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)]
    if significantGenes.empty:
        log.error("There aren't any significant differentially expressed genes")
    else:
        significantCounts = transformedCounts.loc[significantGenes.index].copy(
        )
        pca = PCA()
        pcaComponents = pca.fit_transform(significantCounts.T)
        pcaComponents = pd.DataFrame(
            pcaComponents, index=significantCounts.columns, columns=range(pcaComponents.shape[1]))
        pcaComponents["Group"] = phenotypeData[phenotypeColumn]

        sns.lineplot(x=[f"PC{n+1}" for n in range(phenotypeData.shape[0])],
                     y=np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        if path is None:
            return plt.gca()
        else:
            plt.savefig(path, dpi=300)
            return plt.gca()


def plotPCAComponentsDensity(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                             n_components=None, pValueThreshold=0.05, lfcThreshold=0, path=None):
    if n_components == None:
        n_components = phenotypeData.shape[0]
    elif n_components > phenotypeData.shape[0]:
        raise ValueError(
            "n_components must be lower or equal to the number of samples")
    else:
        LFCColumnName = [col for col in testResults.columns if (
            col.startswith("log2 fold change") and "Intercept" not in col)][0]
        PvalueColumnName = [col for col in testResults.columns if (
            col.startswith("adjusted p-value") and "Intercept" not in col)][0]
        significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
            testResults[PvalueColumnName] < pValueThreshold)]
        if significantGenes.empty:
            log.error(
                "There aren't any significant differentially expressed genes")
        else:
            significantCounts = transformedCounts.loc[significantGenes.index].copy(
            )
            pca = PCA()
            pcaComponents = pca.fit_transform(significantCounts.T)
            pcaComponents = pd.DataFrame(pcaComponents, index=significantCounts.columns, columns=[
                                         f"PC{n+1}" for n in range(pcaComponents.shape[1])])
            pcaComponents["Group"] = phenotypeData[phenotypeColumn]
            for n in range(n_components):
                sns.kdeplot(pcaComponents[f"PC{n+1}"], label=f"PC{n+1}")
            plt.legend()
            plt.xlabel("Principal Components")
            if path is None:
                return plt.gca()
            else:
                plt.savefig(path, dpi=300)
                return plt.gca()


def plotPCA(transformedCounts, testResults, phenotypeData, phenotypeColumn,
            components=("PC1", "PC2"), pValueThreshold=0.05, lfcThreshold=0, path=None):
    if len(components) != 2:
        raise RuntimeError("Provide 2 components in a list, touple or set")
    allComponents = [f"PC{n+1}" for n in range(phenotypeData.shape[0])]
    for comp in components:
        if comp not in allComponents:
            raise RuntimeError(
                f"Component {comp} not found.\nComponents are {allComponents}")
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)]
    if significantGenes.empty:
        log.error("There aren't any significant differentially expressed genes")
    else:
        significantCounts = transformedCounts.loc[significantGenes.index].copy(
        )
        pca = PCA()
        pcaComponents = pca.fit_transform(significantCounts.T)
        pcaComponents = pd.DataFrame(
            pcaComponents, index=significantCounts.columns, columns=allComponents)
        pcaComponents["Group"] = phenotypeData[phenotypeColumn]
        sns.scatterplot(x=components[0], y=components[1],
                        data=pcaComponents, hue="Group", s=50)
        plt.title("PCA")
        if path is None:
            return plt.gca()
        else:
            plt.savefig(path, dpi=300)
            return plt.gca()


def plotIndependentFiltering(results, baseMean, alpha=0.05, method="fdr_bh", path=None):
    PvalueColumnName = [col for col in results.columns if (
        col.startswith("p-value") and "Intercept" not in col)][0]
    pvalues = results[PvalueColumnName].values
    lowerQuantile = np.mean(baseMean == 0)
    upperQuantile = 0.95 if lowerQuantile < 0.95 else 1
    theta = np.linspace(lowerQuantile, upperQuantile, 50)
    cutoffs = np.quantile(baseMean, theta)
    padjMatrix = np.full((len(baseMean), len(cutoffs)), np.NaN)
    for i in range(len(cutoffs)):
        use = baseMean >= cutoffs[i]
        if any(use):
            padjMatrix[use & ~np.isnan(pvalues), i] = multipletests(
                pvalues[use][~np.isnan(pvalues[use])], alpha=alpha, method=method)[1]
    if (np.isnan(padjMatrix)).all():
        return np.full(len(pvalues), np.NaN)

    numRej = np.where(~np.isnan(padjMatrix), padjMatrix
                      < alpha, False).sum(axis=0)
    lowess_fit = sm.nonparametric.lowess(
        numRej, theta, xvals=theta, frac=1/5)
    if max(numRej) <= 10:
        j = 0
    else:
        residual = 0 if all(
            numRej == 0) else numRej[numRej > 0] - lowess_fit[numRej > 0]
        threshold = max(lowess_fit) - np.sqrt(np.mean(residual**2))
        j = np.where(numRej > threshold)[0][0] if any(
            numRej > threshold) else 0
    # padj = padjMatrix[:, j]
    baseMeanThreshold = cutoffs[j]
    filterThreshold = theta[j]
    filterNumRej = numRej[j]

    plt.scatter(theta, numRej, edgecolors="royalblue",
                facecolors="none", alpha=0.8)
    plt.plot(theta, lowess_fit, color="grey", alpha=0.8)
    plt.axvline(filterThreshold, color="firebrick", alpha=0.8, zorder=100)
    plt.text(filterThreshold*1.1, min(numRej)*1.1,
             f"Threshold:\nQuantile={round(filterThreshold,2)}\nNormalized Counts Mean={round(baseMeanThreshold, 3)}\nNumber of Rejections={filterNumRej}", fontsize="x-small", verticalalignment="bottom", zorder=100)
    plt.ylabel("Number of Rejections")
    plt.xlabel("Quantiles of Filter")
    plt.title("Independent Filtering on the Mean of Normalized Counts")

    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotCooks(results, baseMean, path=None):
    PvalueColumnName = [col for col in results.columns if (
        col.startswith("p-value") and "Intercept" not in col)][0]
    pvalues = results[PvalueColumnName].values
    plt.scatter(baseMean+1, -np.log10(pvalues), s=2,
                color="cornflowerblue", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Mean of Normalized Counts")
    plt.ylabel(r"$-log_{10}$(p-value)")

    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300)
        return plt.gca()


def plotCoexpressionNetwork(transformedCounts, testResults, lfcThreshold=1, pValueThreshold=0.01,
                            correlationMethod="spearman", correlationThreshold=None,
                            onlyPositiveCorrelation=False, seed=12345, labels=True, fontSize=1,
                            nodeAlpha=0.9, edgeAlpha=0.4, path=None):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    significantGenes = testResults[(np.abs(testResults[LFCColumnName]) > lfcThreshold) & (
        testResults[PvalueColumnName] < pValueThreshold)]
    if significantGenes.empty:
        raise RuntimeError(
            "There aren't any significant differentially expressed genes")
    else:
        significantCounts = transformedCounts.loc[significantGenes.index].copy(
        ).T
        adjacencyMatrix = significantCounts.corr(
            method=correlationMethod)
        np.fill_diagonal(adjacencyMatrix.values, 0)
        if correlationThreshold:  # specified correlation threshold
            adjacencyMatrix[adjacencyMatrix.abs()
                            <= correlationThreshold] = 0
        elif correlationThreshold is None:
            correlationThreshold = np.quantile(
                adjacencyMatrix.values, 0.975)
            adjacencyMatrix[adjacencyMatrix.abs()
                            <= correlationThreshold] = 0
        if onlyPositiveCorrelation:
            adjacencyMatrix[adjacencyMatrix < 0] = 0

    G = nx.from_pandas_adjacency(adjacencyMatrix)
    edgeWeight = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    nx.set_node_attributes(G, significantGenes[LFCColumnName].to_dict(), "LFC")
    nodeColor = significantGenes[LFCColumnName].values  # LFC
    nx.set_node_attributes(
        G, significantGenes[PvalueColumnName].to_dict(), "FDR")
    # -log10FDR
    if onlyPositiveCorrelation:
        edgeColor = "black"
    else:
        edgeColor = ["firebrick" if weigth
                     > 0 else "royalblue" for weigth in edgeWeight]
    nodeSize = -np.log10(significantGenes[PvalueColumnName].values)
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_edges(G, pos=pos,
                           width=((np.abs(edgeWeight)-np.min(np.abs(edgeWeight)))
                                  / (np.max(np.abs(edgeWeight)-np.min(np.abs(edgeWeight))))),  # weights are rescaled to be within 0 and 1
                           edge_color=edgeColor, alpha=edgeAlpha)
    nx.draw_networkx_nodes(G, pos=pos, node_color=nodeColor,
                           cmap="coolwarm", node_size=nodeSize, alpha=nodeAlpha)
    if labels:
        nx.draw_networkx_labels(G, pos=pos, font_size=fontSize)

    plt.box(False)
    if path is None:
        return plt.gca()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        return plt.gca()
