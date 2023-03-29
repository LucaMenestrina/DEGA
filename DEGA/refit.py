import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from scipy.stats import f
from patsy import dmatrix

from DEGA.utils import getBaseMeansAndVariances, sufficientReplicates, buildResultsDataFrame
from DEGA.dispersions import estimateDispersionsGeneEst, estimateDispersionsMAP
from DEGA.statisticalTests import WaldTest, LRTTest, getMaxCooks
from DEGA.logger import log


def replaceOutliers(counts, normalizedCounts, cooks, designMatrix, sizeFactors, whichSamples=None,
                    cooksCutoff=None, minReplicates=7, trim=0.2, test="Wald"):
    if minReplicates < 3:
        raise ValueError(
            "At least 3 replicates are necessary in order to indentify a sample as a count outlier")
    if cooksCutoff is None:
        cooksCutoff = f.ppf(0.99, designMatrix.shape[1],
                            designMatrix.shape[0] - designMatrix.shape[1])
    idx = np.where(cooks > cooksCutoff)
    replace = (cooks > cooksCutoff).sum(axis=1) > 0
    trimBaseMean = trim_mean(normalizedCounts, trim, axis=1)
    replacementCounts = np.outer(trimBaseMean, sizeFactors).astype("int")

    # replace only those values which fall above the cutoff on Cook's distance
    newCounts = counts.copy()
    newCounts.values[idx] = replacementCounts[idx]
    newNormalizedCounts = newCounts / sizeFactors
    if whichSamples is None:
        whichSamples = sufficientReplicates(designMatrix, minReplicates)

    if whichSamples.sum() == 0:
        return counts, normalizedCounts, replace
    else:
        newCounts.values[:, ~whichSamples] = counts.values[:, ~whichSamples]
        newNormalizedCounts.values[:, ~
                                   whichSamples] = normalizedCounts.values[:, ~whichSamples]
        return newCounts, newNormalizedCounts, replace


def refitWithoutOutliers(counts, normalizedCounts, phenotypeData, cooks, sizeFactors, test,
                         meanVarZero, fullDesignMatrix, reducedDesignMatrix, dispersionsResults,
                         dispFitFunc, testResults, dispPriorVar=None, weights=None, useWeights=False,
                         weightThreshold=1e-2, minDisp=1e-8, kappa0=1, dispTolerance=1e-6,
                         betaTolerance=1e-8, niter=1, linearMu=None, alphaInit=None, fitType="parametric",
                         maxit=100, useOptim=True, forceOptim=False, useT=False, useQR=True, alpha=0.05,
                         pAdjustMethod="fdr_bh", dof=None, betaPrior=False, betaPriorVar=None,
                         betaPriorMethod="weighted", upperQuantile=0.05, useCR=True, minmu=0.5,
                         outlierSD=2, compute_d2log_posterior=False, minReplicatesForReplace=7, cooksCutoff=None):
    allZero = meanVarZero["allZero"].values
    counts, normalizedCounts, replace = replaceOutliers(counts, normalizedCounts, cooks, fullDesignMatrix, sizeFactors,
                                                        whichSamples=None, cooksCutoff=cooksCutoff,
                                                        minReplicates=minReplicatesForReplace)
    nrefit = replace.sum()
    if nrefit > 0:
        meanVarZero = getBaseMeansAndVariances(normalizedCounts, weights)
        allZero = meanVarZero["allZero"].values
        newAllZero = np.where(replace & allZero)[0]
    if nrefit > 0 and nrefit > len(newAllZero):
        refitReplace = replace & ~allZero
        countsSub = counts.iloc[refitReplace]
        normalizedCountsSub = normalizedCounts.iloc[refitReplace]
        if not weights is None:
            weightsSub = weights[refitReplace]
        else:
            weightsSub = None
        meanVarZeroSub = getBaseMeansAndVariances(
            normalizedCountsSub, weightsSub)
        baseMeanSub = meanVarZeroSub["baseMean"].values
        baseVarSub = meanVarZeroSub["baseVar"].values

        dispGeneEstResultsSub, muSub = estimateDispersionsGeneEst(
            genes=countsSub.index,
            # as ascontiguousarray for C faster computations
            counts=np.ascontiguousarray(countsSub),
            normalizedCounts=normalizedCountsSub.values,
            sizeFactors=sizeFactors,
            baseMean=baseMeanSub,
            baseVar=baseVarSub,
            designMatrix=fullDesignMatrix,
            weights=weightsSub,
            useWeights=useWeights,
            weightThreshold=weightThreshold,
            minDisp=minDisp,
            kappa0=kappa0,
            dispTolerance=dispTolerance,
            betaTolerance=betaTolerance,
            maxit=maxit,
            useCR=useCR,
            niter=niter,
            linearMu=linearMu,
            minmu=minmu,
            alphaInit=alphaInit,
            compute_d2log_posterior=compute_d2log_posterior,
        )

        if fitType == "mean":
            useForFit = dispGeneEstResultsSub["dispGeneEst"].values > 10*minDisp
        else:
            useForFit = dispGeneEstResultsSub["dispGeneEst"].values > 100*minDisp
        dispFitSub = dispFitFunc(baseMeanSub)
        dispFitResultsSub = buildResultsDataFrame(
            {"useForFit": useForFit, "dispFit": dispFitSub}, index=countsSub.index, designMatrix=fullDesignMatrix)

        dispMAPResultsSub = estimateDispersionsMAP(
            genes=countsSub.index,
            counts=np.ascontiguousarray(countsSub),
            dispGeneEsts=dispGeneEstResultsSub["dispGeneEst"].values,
            dispFit=dispFitSub,
            mu=muSub,
            designMatrix=fullDesignMatrix,
            dispPriorVar=dispPriorVar,
            weights=weightsSub,
            useWeights=useWeights,
            weightThreshold=weightThreshold,
            outlierSD=outlierSD,
            minDisp=minDisp,
            kappa0=kappa0,
            dispTolerance=dispTolerance,
            maxit=maxit,
            useCR=useCR,
            compute_d2log_posterior=compute_d2log_posterior,
        )

        dispersionsResultsSub = pd.concat(
            [dispGeneEstResultsSub, dispFitResultsSub, dispMAPResultsSub], axis=1)

        if test == "Wald":
            (
                testResultsSub,
                cooksSub,
                betaPriorVarSub
            ) = WaldTest(np.ascontiguousarray(countsSub), np.ascontiguousarray(normalizedCountsSub),
                         sizeFactors, dispersionsResultsSub, meanVarZeroSub, fullDesignMatrix,
                         betaTolerance, weightsSub, useWeights, minmu, maxit, useOptim,
                         forceOptim, useT, useQR, dof, alpha, pAdjustMethod, betaPrior,
                         betaPriorVar, betaPriorMethod, upperQuantile)
        elif test == "LRT":
            if reducedDesignMatrix is None:
                log.warning(
                    "Reduced design matrix not provided. Using formula '~1'")
                reducedDesignMatrix = dmatrix(
                    "1", data=phenotypeData, return_type="dataframe")
            (
                testResultsSub,
                cooksSub,
                betaPriorVarSub
            ) = LRTTest(np.ascontiguousarray(countsSub), np.ascontiguousarray(normalizedCountsSub),
                        sizeFactors, dispersionsResultsSub, meanVarZeroSub, fullDesignMatrix,
                        reducedDesignMatrix, betaTolerance, weightsSub, useWeights, minmu, maxit,
                        useOptim, forceOptim, useT, useQR, alpha, pAdjustMethod)
        if all(replace):
            newMaxCooks = np.full(len(counts), np.NaN)
        else:
            newCooks = cooks.copy()
            newCooks[replace] = 0
            newMaxCooks = getMaxCooks(counts, newCooks, fullDesignMatrix)
            newMaxCooks[replace] = np.NaN

        if nrefit > 0:
            meanVarZero.loc[meanVarZeroSub.index] = meanVarZeroSub
            dispersionsResults.loc[dispersionsResultsSub.index] = dispersionsResultsSub
            testResults.loc[testResultsSub.index] = testResultsSub
            testResults["maximum Cook's distance"] = newMaxCooks

    return meanVarZero, dispersionsResults, testResults, replace, counts
