import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import trim_mean
from scipy.stats import norm, chi2
from scipy.ndimage import gaussian_filter1d as kernel_smoothing
import warnings

from DEGA.cython.estimateDispersion import (
    estimateDispersion,
    estimateDispersionGrid,
    trigamma
)
from DEGA.cython.negativeBinomialGLM import fitNegativeBinomial

from DEGA.utils import buildResultsDataFrame, median_absolute_deviation

from DEGA.logger import log


def estimateDispersionsGeneEst(genes, counts, normalizedCounts, sizeFactors, baseMean,
                               baseVar, designMatrix, weights=None, useWeights=False,
                               weightThreshold=1e-2, minDisp=1e-8, kappa0=1, dispTolerance=1e-6,
                               betaTolerance=1e-8, maxit=100, useCR=True, niter=1, linearMu=None,
                               minmu=0.5, alphaInit=None, compute_d2log_posterior=False):
    if np.log(minDisp/10) <= -30:
        raise RuntimeError(
            "For computational stability, log(minDisp/10) should be above -30")

    if not weights is None:
        weights = np.maximum(weights, 1e-6)

    if alphaInit is None:
        Q, R = np.linalg.qr(designMatrix.values)
        mu = (normalizedCounts @ Q) @ (designMatrix.values @
                                       np.linalg.inv(R)).transpose()
        mu = np.maximum(1, mu)

        roughDisp = np.maximum(
            (((normalizedCounts - mu) ** 2 - mu) / mu ** 2).sum(axis=1)
            / (designMatrix.shape[0] - designMatrix.shape[1]),
            0,
        )
        momentsDisp = (
            baseVar
            - (np.mean(1 / sizeFactors) * baseMean)
        ) / (baseMean ** 2)
        alphas_hat = np.minimum(roughDisp, momentsDisp)
    elif isinstance(alphaInit, int) or isinstance(alphaInit, float):
        alphas_hat = np.tile(alphaInit, counts.shape[0])
    else:
        if len(alphaInit) != counts.shape[0]:
            raise RuntimeError(
                "'alphaInit' lenght not matching number of rows in 'counts'")
        alphas_hat = alphaInit
    # bound the rough estimated alpha between minDisp and maxDisp for numeric stability
    maxDisp = max(10, counts.shape[1])
    alphas_hat = alphas_hat_new = alphaInit = np.minimum(
        np.maximum(minDisp, alphas_hat), maxDisp)

    if not(isinstance(niter, int) and niter >= 0):
        raise RuntimeError("'niter' must be a positive integer")
    if linearMu is None:
        linearMu = len(np.unique(designMatrix, axis=0)
                       ) == designMatrix.shape[1]
        if useWeights:
            linearMu = False

    # iterate between mean and dispersion estimation (niter) times
    fitidx = np.tile(True, counts.shape[0])
    mu = np.zeros_like(counts, dtype="float64")
    dispIter = np.zeros(counts.shape[0], dtype="int32")

    for iter in range(niter):
        if not linearMu:
            fit = fitNegativeBinomial(
                counts[fitidx],  # counts
                normalizedCounts[fitidx],  # normalizedCounts
                sizeFactors[fitidx],  # sizeFactors
                designMatrix,  # designMatrix
                alphas_hat[fitidx],  # dispersions
                betaTolerance,  # betaTolerance
                None,  # lambda_term
                maxit,  # maxit
                True,   # useOptim
                False,  # forceOptim
                True,   # useQR
                weights,  # weights
                useWeights,   # useWeights
                minmu,  # minmu
            )
            fitMu = fit["mu"]
        else:
            Q, R = np.linalg.qr(designMatrix.values)
            m = (normalizedCounts[fitidx] @ Q) @ (designMatrix.values @
                                                  np.linalg.inv(R)).transpose()
            fitMu = m * np.tile(sizeFactors,
                                (normalizedCounts[fitidx].shape[0], 1))
        fitMu = np.maximum(fitMu, minmu)
        mu[fitidx] = fitMu

        estimateResults = estimateDispersion(
            counts,  # counts
            designMatrix.values,  # designMatrix
            mu,  # mu
            np.log(alphas_hat),  # log_alphas
            np.log(alphas_hat),  # log_alphas_prior_means
            1.0,  # log_alpha_prior_sigmasq
            np.log(minDisp / 10),  # min_log_alpha
            kappa0,  # kappa0
            dispTolerance,  # tolerance
            maxit,  # maxit
            False,  # use_prior
            weights,  # weights
            useWeights,  # use_weights
            weightThreshold,  # weight_threshold
            useCR,  # use_CR
            compute_d2log_posterior,  # compute_d2log_posterior
        )

        dispIter[fitidx] = estimateResults["iter"]
        alphas_hat_new[fitidx] = np.minimum(
            np.exp(estimateResults["log_alphas"]), maxDisp)
        last_lp = estimateResults["last_lp"]
        initial_lp = estimateResults["initial_lp"]

        fitidx = np.abs(np.log(alphas_hat_new) - np.log(alphas_hat)) > 0.05
        alphas_hat = alphas_hat_new
        if fitidx.sum() == 0:
            break

    dispGeneEsts = alphas_hat
    if niter == 1:
        noIncrease = last_lp < (initial_lp + initial_lp/1e6)
        dispGeneEsts[noIncrease] = alphaInit[noIncrease]
    convergence = (estimateResults["iter"] < maxit) & np.logical_not(
        estimateResults["iter"] == 1)
    refit = np.logical_not(convergence) & (
        np.exp(estimateResults["log_alphas"]) > minDisp * 10
    )
    if any(refit):
        logAlphasGrid = estimateDispersionGrid(
            counts[refit],  # counts
            designMatrix.values,  # designMatrix
            mu[refit],  # mu
            np.tile(0, np.sum([refit])).astype(float),  # log_alpha_prior_means
            1.0,  # log_alpha_prior_sigmasq
            False,  # use_prior
            weights,  # weights
            useWeights,  # use_weights
            weightThreshold,  # weight_threshold
            useCR,  # use_CR
        )

        gridResults = np.zeros_like(dispGeneEsts, dtype="float64")
        gridResults[refit] = logAlphasGrid
        dispGeneEsts = np.where(refit, gridResults, dispGeneEsts)

    dispGeneEsts = np.minimum(np.maximum(minDisp, dispGeneEsts), maxDisp)

    results = buildResultsDataFrame(
        {"dispGeneEst": dispGeneEsts, "dispGeneIter": dispIter}, index=genes, designMatrix=designMatrix)
    return results, mu


def estimateDispersionsFit(genes, dispGeneEsts, baseMean, designMatrix, fitType="parametric", minDisp=1e-8):
    useForFit = dispGeneEsts > 100*minDisp
    if useForFit.sum() == 0:
        log.warning("All gene-wise dispersion estimates are within 2 orders of magnitude from the minimum value, and so the standard curve fitting techniques will not work.\nGene-wise estimates will be used as dispersions")
        results = buildResultsDataFrame(
            {"useForFit": useForFit, "dispFit": dispGeneEsts}, index=genes, designMatrix=designMatrix)
        fitParams = {"fitType": "dispGeneEsts"}
        return results, fitParams
    else:
        if fitType == "parametric":
            disps = dispGeneEsts[useForFit]
            means = baseMean[useForFit]
            params = np.array([0.1, 1])
            iter = 0
            while iter <= 10:
                residuals = disps / (params[0]+params[1]/means)
                good = np.where(((residuals > 1e-4) & (residuals < 15)))
                if all([idxs.size == 0 for idxs in good]):
                    if iter == 0:
                        log.warning(
                            "All gene-wise dispersion residuals are lower than 1e-4 or higher than 15, and so the parametric fitting technique can't be applied.\nA local regression fit will be attempted")
                        return estimateDispersionsFit(genes, dispGeneEsts, baseMean,
                                                      designMatrix, "local", minDisp)
                    else:
                        break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parametricGLM = sm.GLM(
                        disps[good],
                        sm.add_constant(
                            1 / means[good]) if len(means[good]) > 1 else np.array([[1, means[good][0]]]),
                        family=sm.families.Gamma(
                            link=sm.families.links.identity()),
                    )
                parametric_results = parametricGLM.fit(
                    start_params=params)
                oldParams = params
                params = parametric_results.params
                iter += 1
                if not all(params > 0):
                    log.warning("Parametric dispersion fit failed")
                    parametric_results.converged = False
                elif (np.log(params/oldParams)**2 < 1e-6).all() & parametric_results.converged:
                    break
            else:
                log.warning("Dispersion fit did not converge")
                parametric_results.converged = False
            dispFit = parametricGLM.predict(
                params=parametric_results.params,
                exog=sm.add_constant(1 / baseMean) if len(baseMean) > 1 else np.array([[1, baseMean[0]]]))

            def dispFitFunc(baseMean):
                return parametricGLM.predict(params=parametric_results.params,
                                             exog=sm.add_constant(1 / baseMean) if len(baseMean) > 1 else np.array([[1, baseMean[0]]]))
            fitParams = {"fitType": "parametric"}
            fitParams.update(dict(zip(("asymptDisp", "extraPois"),
                             parametric_results.params)))
            if not parametric_results.converged:
                log.warning("Dispersion trend was not well captured by the function: y = a/x + b, and a local regression fit was automatically substituted.\nspecify fitType='local' or 'mean' to avoid this message next time.")
                return estimateDispersionsFit(genes, dispGeneEsts, baseMean,
                                              designMatrix, "local", minDisp)
        elif fitType == "local":
            subset = dispGeneEsts[useForFit] >= 10*minDisp
            if all(dispGeneEsts[useForFit][subset] < minDisp*10):
                dispFit = np.tile(minDisp, dispGeneEsts[useForFit].shape[0])
            else:
                lowess_fit = sm.nonparametric.lowess(
                    np.log(dispGeneEsts[useForFit][subset]),
                    np.log(baseMean[useForFit][subset]),
                    xvals=np.log(baseMean)
                )
                dispFit = np.exp(lowess_fit)

            def dispFitFunc(baseMeanToFit):
                return np.exp(sm.nonparametric.lowess(np.log(dispGeneEsts[useForFit][subset]),
                                                      np.log(
                                                          baseMean[useForFit][subset]),
                                                      xvals=np.log(baseMeanToFit)))
            fitParams = {"fitType": "local"}
        elif fitType == "mean":
            useForFit = dispGeneEsts > 10 * minDisp
            mean = trim_mean(
                dispGeneEsts[useForFit], proportiontocut=0.001
            )
            meanFit = np.repeat(
                mean,
                len(baseMean),
            )

            def dispFitFunc(baseMean):
                return np.repeat(mean, len(baseMean))
            dispFit = meanFit
            fitParams = {"fitType": "mean", "mean": mean}
        elif fitType == "kernel":
            kernelFitPrep = kernel_smoothing(
                dispGeneEsts, np.std(dispGeneEsts))
            # for smoothing out outliers
            dispGeneEstsForFit = np.where(
                useForFit, dispGeneEsts, kernelFitPrep)
            kernelFit = kernel_smoothing(
                dispGeneEstsForFit, np.std(dispGeneEstsForFit))

            def dispFitFunc(baseMean):
                kernelFitPrep = kernel_smoothing(
                    dispGeneEsts, np.std(dispGeneEsts))
                # for smoothing out outliers
                dispGeneEstsForFit = np.where(
                    useForFit, dispGeneEsts, kernelFitPrep)
                kernelFit = kernel_smoothing(
                    dispGeneEstsForFit, np.std(dispGeneEstsForFit))
                return kernelFit
            dispFit = kernelFit
            fitParams = {"fitType": "kernel"}
        else:
            raise RuntimeError(f"'fitType' {fitType} not implemented")

    results = buildResultsDataFrame(
        {"useForFit": useForFit, "dispFit": dispFit}, index=genes, designMatrix=designMatrix)
    return results, dispFitFunc, fitParams


def estimateDispersionsPriorVar(dispGeneEsts, dispFit, designMatrix, minDisp=1e-8):
    aboveMinDisp = dispGeneEsts >= minDisp * 100
    dispResiduals = np.log(dispGeneEsts) - np.log(dispFit)
    if aboveMinDisp.sum() == 0:
        raise RuntimeError("No data found which is greater than minDisp")
    varLogDispEsts = median_absolute_deviation(
        dispResiduals[aboveMinDisp]) ** 2

    m, p = designMatrix.shape
    if (m - p) <= 3 and (m > p):
        # set seed (works also for scipy)
        np.random.seed(12345)
        obsDist = dispResiduals[aboveMinDisp]
        breaks = np.arange(-9.5, 10, 0.5)
        obsDist = obsDist[(obsDist > -10) & (obsDist < 10)]
        obsVarGrid = np.lispace(0, 8, 200)
        obsDistHistDensity, _ = np.histogram(obsDist, breaks, density=True)

        def tmp_func(x):
            randDist = np.log(chi2.rvs(m-p, 0, 1, int(1e4))) + \
                norm.rvs(0, np.sqrt(x), int(1e4)) - np.log(m-p)
            randDist = randDist[(randDist > -10) & (randDist < 10)]
            randDistHistDensity, _ = np.histogram(
                randDist, breaks, density=True)
            z = (obsDistHistDensity, randDistHistDensity)
            small = min(z[z > 0])
            kl = (obsDistHistDensity * (np.log(obsDistHistDensity
                  + small)-np.log(randDistHistDensity+small))).sum()
            return kl
        tmp_func = np.vectorize(tmp_func)
        klDivs = tmp_func(obsVarGrid)

        obsVarFineGrid = np.lispace(0, 8, 1000)
        lowess_fit = sm.nonparametric.lowess(
            klDivs, obsVarGrid, xvals=obsVarFineGrid
        )  # frac=0.2 may lead to instabilities

        minKL = obsVarFineGrid[np.argmin(lowess_fit)]
        expVarLogDisp = trigamma((m - p) / 2)
        dispPriorVar = np.maximum(minKL, 0.25)

        return dispPriorVar
    elif m > p:
        expVarLogDisp = trigamma((m - p) / 2)
        dispPriorVar = np.maximum((varLogDispEsts - expVarLogDisp), 0.25)
    else:
        dispPriorVar = varLogDispEsts
        expVarLogDisp = 0

    return dispPriorVar, varLogDispEsts


def estimateDispersionsMAP(genes, counts, dispGeneEsts, dispFit, mu, designMatrix,
                           dispPriorVar=None, weights=None, useWeights=False, weightThreshold=1e-2,
                           outlierSD=2, minDisp=1e-8, kappa0=1.0, dispTolerance=1e-6, maxit=100,
                           useCR=True, compute_d2log_posterior=False):
    if not weights is None:
        weights = np.maximum(weights, 1e-6)

    if dispPriorVar is None:
        if all(dispGeneEsts < minDisp*100):
            log.warning(
                f"All genes have dispersion estimates < {minDisp*10}, returning disp = {minDisp*10}")
            results = buildResultsDataFrame(
                {"dispersion": np.tile(minDisp*10, counts.shape[0])}, index=genes, designMatrix=designMatrix)

            return results
        else:
            dispPriorVar, varLogDispEsts = estimateDispersionsPriorVar(
                dispGeneEsts, dispFit, designMatrix.values, minDisp=minDisp)
    else:
        aboveMinDisp = dispGeneEsts >= minDisp * 100
        dispResiduals = np.log(dispGeneEsts) - np.log(dispFit)
        varLogDispEsts = median_absolute_deviation(
            dispResiduals[aboveMinDisp]) ** 2

    # use estimate gene dispersion if at least one order of magnitute greater than the fitted line, else use the fitted line
    dispInit = np.where(
        dispGeneEsts > 0.1 * dispFit, dispGeneEsts, dispFit
    )
    # if any missing values, fill in the fitted value to initialize
    dispInit[np.isnan(dispInit)] = dispFit[np.isnan(dispInit)]

    estimateResults = estimateDispersion(
        counts,  # counts
        designMatrix.values,  # designMatrix
        mu,  # mu
        np.log(dispInit),  # log_alphas
        np.log(dispFit),  # log_alphas_prior_means
        dispPriorVar,  # log_alpha_prior_sigmasq
        np.log(minDisp / 10),  # min_log_alpha
        kappa0,  # kappa0
        dispTolerance,  # tolerance
        maxit,  # maxit
        True,  # use_prior
        weights,  # weights
        useWeights,  # use_weights
        weightThreshold,  # weight_threshold
        useCR,  # use_CR
        compute_d2log_posterior,  # compute_d2log_posterior
    )

    dispMAP = np.exp(estimateResults["log_alphas"])
    dispIter = estimateResults["iter"]

    convergence = (estimateResults["iter"] < maxit)
    refit = np.logical_not(convergence)
    if any(refit):
        logAlphasGrid = estimateDispersionGrid(
            counts[refit],  # counts
            designMatrix.values,  # designMatrix
            mu[refit],  # mu
            np.tile(0, np.sum([refit])).astype(float),  # log_alpha_prior_means
            1.0,  # log_alpha_prior_sigmasq
            False,  # use_prior
            weights,  # weights
            useWeights,  # use_weights
            weightThreshold,  # weight_threshold
            useCR,  # use_CR
        )

        gridResults = np.zeros_like(dispMAP, dtype="float64")
        gridResults[refit] = logAlphasGrid
        dispMAP = np.where(refit, gridResults, dispMAP)

    maxDisp = max(10, counts.shape[1])
    dispMAP = np.minimum(np.maximum(minDisp, dispMAP), maxDisp)

    dispFinal = dispMAP.copy()
    dispOutlier = np.log(dispGeneEsts) > (
        np.log(dispFit) + outlierSD * np.sqrt(varLogDispEsts))
    dispFinal[dispOutlier] = dispGeneEsts[dispOutlier]

    results = buildResultsDataFrame({"dispMAP": dispMAP,
                                     "dispIter": dispIter,
                                     "dispOutlier": dispOutlier,
                                     "dispersion": dispFinal
                                     }, index=genes, designMatrix=designMatrix)
    return results


def estimateDispersions(counts, normalizedCounts, sizeFactors, meanVarZero, designMatrix,
                        dispPriorVar=None, weights=None, useWeights=False, weightThreshold=1e-2,
                        minDisp=1e-8, kappa0=1, dispTolerance=1e-6, betaTolerance=1e-6, niter=1,
                        linearMu=None, alphaInit=None, fitType="parametric", maxit=100,
                        useCR=True, minmu=0.5, outlierSD=2, compute_d2log_posterior=False):
    baseMean = meanVarZero["baseMean"].values
    baseVar = meanVarZero["baseVar"].values
    # allZero = meanVarZero["allZero"].values
    if designMatrix.shape[0] == designMatrix.shape[1]:
        raise RuntimeError(
            "The design matrix has the same number of samples and coefficients to fit, so estimation of dispersion is not possible.")
    dispGeneEstResults, mu = estimateDispersionsGeneEst(
        genes=counts.index,
        # as ascontiguousarray for C faster computations
        counts=np.ascontiguousarray(counts),
        normalizedCounts=normalizedCounts.values,
        sizeFactors=sizeFactors,
        baseMean=baseMean,
        baseVar=baseVar,
        designMatrix=designMatrix,
        weights=weights,
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

    dispFitResults, dispFitFunc, fitParams = estimateDispersionsFit(
        genes=counts.index,
        dispGeneEsts=dispGeneEstResults["dispGeneEst"].values,
        baseMean=baseMean,
        designMatrix=designMatrix,
        fitType=fitType,
        minDisp=minDisp
    )
    if dispFitResults["useForFit"].values.sum() == 0:
        # returns dispersions = dispGeneEsts
        return (buildResultsDataFrame({"dispersion": dispGeneEstResults["dispGeneEst"].values},
                                      index=counts.index, designMatrix=designMatrix),
                dispFitFunc, fitParams, mu)
    else:
        dispMAPResults = estimateDispersionsMAP(
            genes=counts.index,
            counts=np.ascontiguousarray(counts),
            dispGeneEsts=dispGeneEstResults["dispGeneEst"].values,
            dispFit=dispFitResults["dispFit"].values,
            mu=mu,
            designMatrix=designMatrix,
            dispPriorVar=dispPriorVar,
            weights=weights,
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
        return pd.concat([dispGeneEstResults, dispFitResults, dispMAPResults], axis=1), dispFitFunc, fitParams, mu
