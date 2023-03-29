import numpy as np
import pandas as pd
from scipy.stats import f, t, norm, chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from DEGA.cython.negativeBinomialGLM import fitNegativeBinomial, fitNegativeBinomialWithPrior

from DEGA.utils import buildResultsDataFrame, sufficientReplicates, trimmedVariance, trimmedSampleVariance

from DEGA.logger import log


def robustMethodOfMomenthsDisp(normalizedCounts, designMatrix, minReplicates=3):
    samples, sample_level, sample_count = np.unique(
        designMatrix.values, return_counts=True, return_inverse=True, axis=0)
    validSamples = (sample_count >= minReplicates)[sample_level]
    if any(validSamples):
        _, valid_sample_level, valid_sample_count = np.unique(
            sample_level[validSamples], return_counts=True, return_inverse=True)
        variance = trimmedSampleVariance(
            normalizedCounts[:, validSamples], valid_sample_level)
    else:
        variance = trimmedVariance(normalizedCounts)

    mean = normalizedCounts.mean(axis=1)
    alpha = (variance - mean)/(mean**2)
    # cannot use the typical minDisp = 1e-8 here or else
    # all counts in the same group as the outlier count will get an extreme Cook's distance
    minDisp = 0.04
    alpha = np.maximum(alpha, minDisp)

    return alpha


def CooksDistance(counts, normalizedCounts, mu, diagonals, designMatrix, minReplicates=3):
    disps = robustMethodOfMomenthsDisp(
        normalizedCounts, designMatrix, minReplicates=minReplicates)
    V = mu + (disps.reshape(-1, 1) * mu**2)
    Pearson = ((counts-mu)**2)/V
    cooks = Pearson / \
        designMatrix.shape[1] * \
        diagonals / (1-diagonals)**2

    return cooks


def getMaxCooks(counts, cooks, designMatrix, minReplicates=3):
    validSamples = sufficientReplicates(designMatrix, minReplicates)

    if (designMatrix.shape[0] > designMatrix.shape[1]) & any(validSamples):
        maxCooks = np.max(cooks[:, validSamples], axis=1)
    else:
        maxCooks = np.tile(None, counts.shape[0])

    return maxCooks


def adjustPValue(pvalues, baseMean, alpha=0.05, method="fdr_bh"):
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
    padj = padjMatrix[:, j]
    # baseMeanThreshold = cutoffs[j]
    # filterThreshold = theta[j]
    # filterNumRej = numRej[j]

    return padj


def WaldTest(counts, normalizedCounts, sizeFactors, dispersionsResults, meanVarZero,
             designMatrix, betaTolerance=1e-8, weights=None, useWeights=False, minmu=0.5,
             maxit=100, useOptim=True, forceOptim=False, useT=False, useQR=True, dof=None,
             alpha=0.05, pAdjustMethod="fdr_bh", betaPrior=False, betaPriorVar=None,
             betaPriorMethod="weighted", upperQuantile=0.05):
    dispersions = dispersionsResults["dispersion"].values
    baseMean = meanVarZero["baseMean"].values
    allZero = meanVarZero["allZero"].values
    if betaPrior:
        priorFitResults = fitNegativeBinomialWithPrior(
            counts,  # counts
            normalizedCounts,  # normalizedCounts
            sizeFactors,  # sizeFactors
            designMatrix,  # designMatrix
            dispersionsResults,  # dispersionsResults
            baseMean,  # baseMean
            betaTolerance,  # betaTolerance
            None,  # lambda_term
            maxit,  # maxit
            useOptim,   # useOptim
            forceOptim,  # forceOptim
            useQR,   # useQR
            weights,  # weights
            useWeights,   # useWeights
            betaPriorVar,  # betaPriorVar
            None,  # mu
            None,  # MLE_betaMatrix
            minmu,  # minmu
            betaPriorMethod,   # "weighted"
            upperQuantile,  # 0.05
        )
        fitResults = priorFitResults["fit"]
        diagonals = priorFitResults["diagonals"]
        mu = priorFitResults["mu"]
        betaPriorVar = priorFitResults["betaPriorVar"]
        MLE_beta_matrix = priorFitResults["MLE_betaMatrix"]
    else:
        fitResults = fitNegativeBinomial(
            counts,  # counts
            normalizedCounts,  # normalizedCounts
            sizeFactors,  # sizeFactors
            designMatrix,  # designMatrix
            dispersions,  # dispersions
            betaTolerance,  # betaTolerance
            None,  # lambda_term
            maxit,  # maxit
            useOptim,   # useOptim
            forceOptim,  # forceOptim
            useQR,   # useQR
            weights,  # weights
            useWeights,   # useWeights
            minmu,  # minmu
        )
        diagonals = fitResults["diagonals"]
        mu = fitResults["mu"]
        betaPriorVar = np.tile(1E6, designMatrix.shape[1])

    cooks = CooksDistance(counts, normalizedCounts,
                          mu, diagonals, designMatrix)

    maxCooks = getMaxCooks(counts, cooks, designMatrix)

    betaMatrix = fitResults["betaMatrix"]
    betaSE = fitResults["betaSE"]
    WaldStatistics = betaMatrix/betaSE

    if useT:
        if not dof is None:
            if len(dof) == 1:
                dof = np.tile(dof, counts.shape[0])
            elif len(dof) != counts.shape[0]:
                raise RuntimeError(
                    f"Length of dof is {len(dof)}, but it should match the number of genes with non-zero read counts {counts.shape[0]}")
        else:
            if not weights is None:
                n_samples = np.sum(weights, axis=1)
            else:
                n_samples = np.tile(counts.shape[1], counts.shape[0])
            dof = n_samples - designMatrix.shape[1]

        dof = np.where(dof > 0, dof, None)
        WaldPValue = 2*(t.sf(np.abs(WaldStatistics), dof))
    else:
        WaldPValue = 2*(norm.sf(np.abs(WaldStatistics)))

    adjustedPValue = np.apply_along_axis(lambda x: adjustPValue(
        x, baseMean, alpha=alpha, method=pAdjustMethod), axis=0, arr=WaldPValue)
    betaConv = fitResults["betaConv"]

    if not betaConv.all():
        raise RuntimeError(
            "Some rows did not converge in beta. Use larger maxit argument with WaldTest")

    mleBetas = MLE_beta_matrix if betaPrior else None
    mleInfo = "MLE" if betaPrior else None
    Tdof = dof if useT else None
    lfcType = "MAP" if betaPrior else "MLE"
    tDFDescription = "t degrees of freedom for Wald test" if useT else None

    results = buildResultsDataFrame({
        f"log2 fold change ({lfcType})": betaMatrix,
        "standard error": betaSE,
        mleInfo: mleBetas,
        "statistics": WaldStatistics,
        "p-value": WaldPValue,
        "adjusted p-value": adjustedPValue,
        "convergence of betas": betaConv,
        "iterations of beta": fitResults["betaIter"],
        "deviance from the fitted model": -2 * fitResults["logLike"],
        "maximum Cook's distance": maxCooks,
        tDFDescription: Tdof

    }, index=dispersionsResults.index, designMatrix=designMatrix, NArows=allZero)

    return results, cooks, betaPriorVar


def LRTTest(counts, normalizedCounts, sizeFactors, dispersionsResults, meanVarZero,
            designMatrix, reducedDesignMatrix, betaTolerance=1e-8, weights=None,
            useWeights=False, minmu=0.5, maxit=100, useOptim=True, forceOptim=False,
            useT=False, useQR=True, alpha=0.05, pAdjustMethod="fdr_bh"):
    dispersions = dispersionsResults["dispersion"].values
    baseMean = meanVarZero["baseMean"].values
    allZero = meanVarZero["allZero"].values
    df = designMatrix.shape[1] - reducedDesignMatrix.shape[1]
    if df < 1:
        log.warning(
            "Less than one degree of freedom, perhaps full and reduced models are inverted")

    fitFull = fitNegativeBinomial(
        counts,  # counts
        normalizedCounts,  # normalizedCounts
        sizeFactors,  # sizeFactors
        designMatrix,  # designMatrix
        dispersions,  # dispersions
        betaTolerance,  # betaTolerance
        None,  # lambda_term
        maxit,  # maxit
        useOptim,   # useOptim
        forceOptim,  # forceOptim
        useQR,   # useQR
        weights,  # weights
        useWeights,   # useWeights
        minmu,  # minmu
    )
    fitReduced = fitNegativeBinomial(
        counts,  # counts
        normalizedCounts,  # normalizedCounts
        sizeFactors,  # sizeFactors
        reducedDesignMatrix,  # designMatrix
        dispersions,  # dispersions
        betaTolerance,  # betaTolerance
        None,  # lambda_term
        maxit,  # maxit
        useOptim,   # useOptim
        forceOptim,  # forceOptim
        useQR,   # useQR
        weights,  # weights
        useWeights,   # useWeights
        minmu,  # minmu
    )

    LRTStatistics = 2 * (fitFull["logLike"] - fitReduced["logLike"])
    LRTPvalue = chi2.sf(LRTStatistics, df=df)
    adjustedPValue = np.apply_along_axis(lambda x: adjustPValue(
        x, baseMean, alpha=alpha, method=pAdjustMethod), axis=0, arr=LRTPvalue)
    deviance = -2 * fitFull["logLike"]
    mu = fitFull["mu"]
    diagonals = fitFull["diagonals"]

    cooks = CooksDistance(counts, normalizedCounts,
                          mu, diagonals, designMatrix)
    maxCooks = getMaxCooks(counts, cooks, designMatrix)

    betaPriorVar = np.tile(1E6, designMatrix.shape[1])

    if not fitFull["betaConv"].all():
        raise RuntimeError(
            "Some rows did not converge in beta. Use larger maxit argument with WaldTest")

    results = buildResultsDataFrame({
        "log2 fold change (MLE)": fitFull["betaMatrix"],
        "standard error": fitFull["betaSE"],
        "statistics (full vs reduced)": LRTStatistics,
        "p-value (full vs reduced)": LRTPvalue,
        "adjusted p-value": adjustedPValue,
        "convergence of betas for full design": fitFull["betaConv"],
        "convergence of betas for reduced design": fitReduced["betaConv"],
        "iterations for betas for full design": fitFull["betaIter"],
        "deviance of the full design": deviance,
        "maximum Cook's distance": maxCooks
    }, index=dispersionsResults.index, designMatrix=designMatrix, NArows=allZero)

    return results, cooks, betaPriorVar


def adjustForOutliers(counts, phenotypeData, test, testResults, designMatrix, betaPriorVar,
                      baseMean, cooks, replace, cooksCutoff=None, alpha=0.05, lfcThreshold=1,
                      useT=False, pAdjustMethod="fdr_bh", altHypothesis="greaterAbs"):
    LFCColumnName = [col for col in testResults.columns if (
        col.startswith("log2 fold change") and "Intercept" not in col)][0]
    PvalueColumnName = [col for col in testResults.columns if (
        col.startswith("p-value") and "Intercept" not in col)][0]
    AdjPvalueColumnName = [col for col in testResults.columns if (
        col.startswith("adjusted p-value") and "Intercept" not in col)][0]
    StatColumnName = [col for col in testResults.columns if (
        col.startswith("statistics") and "Intercept" not in col)][0]
    SEColumnName = [col for col in testResults.columns if (
        col.startswith("standard error") and "Intercept" not in col)][0]

    # for readability
    pvalue = testResults[PvalueColumnName].values
    stat = testResults[StatColumnName].values
    LFC = testResults[LFCColumnName].values
    T = lfcThreshold
    SE = testResults[SEColumnName].values

    if test == "Wald" and lfcThreshold != 0:
        if useT:
            dof = testResults["t degrees of freedom for Wald test"]
            def pfunc(x): return t.sf(x, dof)
        else:
            def pfunc(x): return norm.sf(x)
        if altHypothesis == "greaterAbs":
            newStat = np.sign(T) * np.maximum((np.abs(LFC) - T)/SE, 0)
            newPvalue = np.minimum(2 * pfunc((np.abs(LFC) - T)/SE), 1)
        elif altHypothesis == "lessAbs":
            newStatAbove = np.maximum((T-LFC)/SE, 0)
            newPvalueAbove = pfunc((T-LFC)/SE)
            newStatBelow = np.maximum((LFC+T)/SE, 0)
            newPvalueBelow = pfunc((LFC+T)/SE)
            newStat = np.minimum(newStatAbove, newStatBelow)
            newPvalue = np.maximum(newPvalueAbove, newPvalueBelow)
        elif altHypothesis == "greater":
            newStat = np.maximum((T-LFC)/SE, 0)
            newPvalue = pfunc((T-LFC)/SE)
        elif altHypothesis == "less":
            newStat = np.minimum((LFC+T)/SE, 0)
            newPvalue = pfunc((-T-LFC)/SE)
        pvalue = newPvalue
        stat = newStat

    if cooksCutoff is None:
        cooksCutoff = f.ppf(
            0.99, designMatrix.shape[1], designMatrix.shape[0] - designMatrix.shape[1])

    cooksOutlier = testResults["maximum Cook's distance"].values > cooksCutoff
    designVars = designMatrix.columns[designMatrix.columns != "Intercept"]

    if cooksOutlier[~np.isnan(cooksOutlier)].any():
        designVars = designMatrix.columns[designMatrix.columns != "Intercept"]
        if designVars.size == 1 and pd.unique(phenotypeData[designVars].squeeze()).size == 2:
            dontFilter = np.full(
                cooksOutlier[~np.isnan(cooksOutlier)].sum(), False)
            for i in range(dontFilter.size):
                ii = np.where(cooksOutlier)[0][i]
                outCount = counts.values[ii, np.argmax(cooks[ii, :])]

                if (counts.iloc[ii, :] > outCount).sum() >= 3:
                    dontFilter[i] = True

            cooksOutlier[np.where(cooksOutlier)][dontFilter] = False

    if replace[~np.isnan(replace)].sum() > 0:
        nowZero = np.where(replace & (baseMean == 0))
        LFC[nowZero] = 0
        SE[nowZero] = 0
        stat[nowZero] = 0
        pvalue[nowZero] = 1

    pvalue[np.where(cooksOutlier)] = np.NaN

    adjustedPValue = adjustPValue(
        pvalue, baseMean, alpha=alpha, method=pAdjustMethod)

    testResults[LFCColumnName] = LFC
    testResults[SEColumnName] = SE
    testResults[StatColumnName] = stat
    testResults[PvalueColumnName] = pvalue
    testResults[AdjPvalueColumnName] = adjustedPValue
    testResults["Cook's outlier"] = replace

    return testResults
