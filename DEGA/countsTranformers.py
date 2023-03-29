import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import splrep, splev

from DEGA.cython.negativeBinomialGLM import fitNegativeBinomial, matchUpperQuantileForVariance

from DEGA.logger import log


def shiftedLog(normalizedCounts, shift=1):
    return np.log2(normalizedCounts + shift)


def vst(normalizedCounts, sizeFactors, baseMean, dispGeneEsts, fitType, fitParams, minDisp=1e-8, fitFunc=None):
    if fitType == "parametric":
        extraPois = fitParams["extraPois"]
        asymptDisp = fitParams["asymptDisp"]
        scaled = np.log((1 + extraPois
                         + 2*asymptDisp*normalizedCounts
                         + 2*np.sqrt(asymptDisp * normalizedCounts
                                     * (1 + extraPois + asymptDisp*normalizedCounts))
                         ) / (4*asymptDisp)) / np.log(2)
        scaled = pd.DataFrame(
            scaled, index=normalizedCounts.index, columns=normalizedCounts.columns)
        return scaled
    elif fitType == "local":
        useForFit = dispGeneEsts >= 10*minDisp
        xg = np.sinh(np.linspace(np.arcsinh(0), np.arcsinh(
            np.max(normalizedCounts.values)), 1000))[1:]
        xim = np.mean(1/sizeFactors)
        xgf = fitFunc(xg)
        baseVarsAtGrid = xgf * xg**2 + xim * xg
        integrand = 1 / np.sqrt(baseVarsAtGrid)
        splf = splrep(np.arcsinh((xg[1:] + xg[:-1])/2),
                      np.cumsum((xg[1:] + xg[:-1])*(integrand[:1] + integrand[:-1])/2))
        h1 = np.quantile(normalizedCounts.mean(axis=1), 0.95)
        h2 = np.quantile(normalizedCounts.mean(axis=1), 0.999)
        eta = (np.log2(h2)-np.log2(h1)) / \
            (splev(np.arcsinh(h2), splf) - splev(np.arcsinh(h1), splf))
        xi = np.log2(h1) - eta*splev(np.arcsinh(h1), splf)
        scaled = np.apply_along_axis(lambda col: eta * splev(np.arcsinh(col), splf) + xi,
                                     axis=0, arr=normalizedCounts)
        scaled = pd.DataFrame(
            scaled, index=normalizedCounts.index, columns=normalizedCounts.columns)
        return scaled
    elif fitType == "mean":
        scaled = (2 * np.arcsinh(np.sqrt(fitParams["mean"] * normalizedCounts)) - np.log(
            fitParams["mean"]) - np.log(4))/np.log(2)
        scaled = pd.DataFrame(
            scaled, index=normalizedCounts.index, columns=normalizedCounts.columns)
        return scaled
    else:
        raise RuntimeError(
            "vst only available for fitType: 'parametric', 'local', 'mean'")


def sparsityTest(normalizedCounts, p, t1, t2):
    rowSums = np.sum(normalizedCounts, axis=1)
    if any(rowSums > t1):
        rowMax = np.max(normalizedCounts, axis=1)
        prop = (rowMax/rowSums)[rowSums > t1]
        total = np.mean(prop[prop > p])
        if total > t2:
            log.warning(
                f"rlog assumes that data is close to a negative binomial distribution, an assumption which is sometimes not compatible with datasets where many genes have many zero counts despite a few very large counts. In this data, for {round(total,3)*100}% of genes with a sum of normalized counts above {t1}, it was the case that a single sample's normalized count made up more than {p*100}% of the sum over all samples. the threshold for this warning is {t2*100}% of genes. See the 'sparsityPlot' for a visualization of this. It is recommend instead using the variance stabilizing transformation (vst) or shifted log (shiftedLog).")


def rlog(counts, normalizedCounts, sizeFactors, baseMean, dispFit, intercept=None, betaPriorVar=None):
    tcounts = counts.copy()
    tnormalizedCounts = normalizedCounts.copy()
    n_samples = tnormalizedCounts.shape[1]
    if (n_samples > 10 and n_samples < 30):
        log.warning(
            "rlog may take a few minutes with more than 10 samples, vst is a much faster transformation")
    elif n_samples >= 30:
        log.warning(
            "rlog may take a long time with 30 or more samples, vst is a much faster transformation")

    if intercept is None:
        sparsityTest(tnormalizedCounts, 0.9, 100, 0.1)

        designMatrix = pd.DataFrame(np.concatenate((np.ones((n_samples, 1)), np.eye(
            n_samples)), axis=1), columns=["Intercept"]+[f"sample{n+1}" for n in range(n_samples)])
    else:
        if len(intercept) != tnormalizedCounts.shape[0]:
            raise RuntimeError(
                "Intercept should be as long as the number of rows of object")
        if n_samples == 1:
            designMatrix = pd.DataFrame([1], index=[1], columns=["sample1"])
        else:
            designMatrix = pd.DataFrame(np.eye(
                n_samples+1), columns=["sample_null_level"]+[f"sample{n+1}" for n in range(n_samples)])
        infiniteIntercept = ~np.isfinite(intercept)
        tcounts = tcounts.loc[infiniteIntercept, :]
        intercept[infiniteIntercept] = -10
        tnormalizedCounts = tcounts/(sizeFactors * 2**intercept)

    if betaPriorVar is None:
        # if a prior sigma squared not provided,
        # estimate it by the matching upper quantiles of the log2 counts plus a pseudocount of 0.5
        logFoldChange = np.log2(tnormalizedCounts+0.5) - \
                                np.log2(baseMean+0.5).reshape(-1, 1)
        weights = 1/((1/baseMean) + dispFit)
        betaPriorVar = matchUpperQuantileForVariance(
             logFoldChange.values, 0.05, weights)

    lambda_term = 1/np.tile(betaPriorVar, designMatrix.shape[1])
    if "Intercept" in designMatrix.columns:
        lambda_term[np.where(designMatrix.columns == "Intercept")[0]] = 1e-6
    fit = fitNegativeBinomial(
        np.ascontiguousarray(tcounts),  # counts
        np.ascontiguousarray(tnormalizedCounts),  # normalizedCounts
        sizeFactors,  # sizeFactors
        designMatrix,  # designMatrix
        dispFit,  # dispersions
        1e-4,  # betaTolerance
        lambda_term,  # lambda_term
        100,  # maxit
        False,   # useOptim
        False,  # forceOptim
        True,   # useQR
        None,  # weights
        False,   # useWeights
        0.5,  # minmu
    )

    scaled = (designMatrix.values @ fit["betaMatrix"].T).T

    if not intercept is None:
        scaled += np.where(infiniteIntercept, 0, intercept)

    return pd.DataFrame(scaled, index=normalizedCounts.index, columns=normalizedCounts.columns)
