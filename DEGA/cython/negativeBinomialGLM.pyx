# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

# cython: np_pythran=True
# cython: cxx=True
# cython: -DUSE_XSIMD -fopenmp -march=native

import numpy as np
from scipy.stats import nbinom, norm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from libc.math cimport log, exp  # only on scalars

import cython
cimport numpy as np
np.import_array()

cdef DOUBLE = np.double

cdef fitBeta(
        counts,
        designMatrix,
        sizeFactors,
        dispersions,
        beta_matrix,
        lambda_term,
        weights,
        useWeights,
        betaTolerance,
        maxit,
        useQR,
        minmu
        ):

    n_rows = len(counts)  # counts.shape[0]
    n_samples = designMatrix.shape[0]
    n_variables = designMatrix.shape[1]

    iter = np.zeros(n_rows, dtype=np.int32)
    beta_var_matrix = np.zeros_like(beta_matrix, dtype=np.float64, order="C")
    contrast = np.concatenate(
        ([1], np.zeros(n_variables-1, dtype=np.float64, order="C")))
    contrast_numerator = np.zeros(
        (len(beta_matrix), 1), dtype=np.float64, order="C")
    contrast_denominator = np.zeros(
        (len(beta_matrix), 1), dtype=np.float64, order="C")
    diagonals = np.zeros_like(counts, dtype=np.float64, order="C")
    deviances = np.zeros(n_rows, dtype=np.float64, order="C")

    ridge = np.diag(lambda_term)

    size = 1/dispersions

    mu = np.maximum(
        sizeFactors * np.exp(np.matmul(designMatrix, beta_matrix.T)).T, minmu)

    toFit = np.ones(n_rows, dtype=bool, order="C")  # np.tile(True, n_rows)

    if useQR:
        for i in range(maxit):
            iter[toFit] += 1
            if useWeights:
                w = weights[toFit] * \
                    (mu[toFit]/(1+(dispersions[toFit]*mu[toFit].T)).T)
                w_sqrt = np.sqrt(w)
            else:
                w = mu[toFit]/(1+(dispersions[toFit]*mu[toFit].T)).T
                w_sqrt = np.sqrt(w)

            weighted_x_ridge = np.apply_along_axis(lambda row: np.concatenate((np.multiply(
                designMatrix, row.reshape(-1, 1)), np.sqrt(ridge))), arr=w_sqrt, axis=1)
            Q, R = zip(*[np.linalg.qr(mat) for mat in weighted_x_ridge])
            Q, R = np.array(Q), np.array(R)
            z = np.log(mu[toFit]/sizeFactors) + \
                ((counts[toFit] - mu[toFit])/mu[toFit])
            z_sqrt_w = (z * w_sqrt)
            big_z_sqrt_w = np.zeros((n_rows, n_samples+n_variables))
            big_z_sqrt_w[toFit, 0:n_samples] = z_sqrt_w
            beta = np.array([np.linalg.solve(R[i], Q[i].T @ big_z_sqrt_w[toFit][i])
                            for i in range(toFit.sum())])
            beta_matrix[toFit] = beta

            mu = np.maximum(
                sizeFactors * np.exp(np.matmul(designMatrix, beta_matrix.T)).T, minmu)

            deviance = np.zeros(toFit.sum())

            for sample in range(n_samples):
                if useWeights:
                    deviance = deviance + (-2 * weights[toFit, sample] * np.log(nbinom.pmf(
                        k=counts[toFit, sample], n=size[toFit], p=size[toFit]/(size[toFit]+mu[toFit, sample]))))
                else:
                    deviance = deviance + \
                        (-2 * np.log(nbinom.pmf(k=counts[toFit, sample],
                         n=size[toFit], p=size[toFit]/(size[toFit]+mu[toFit, sample]))))

            convergence = np.abs(
                deviance-deviances[toFit])/(np.abs(deviance) + 0.1)

            deviances[toFit] = deviance
            if i > 0:
                toFit[toFit] = (convergence >= betaTolerance)
                if all(toFit == False):
                    break

            br = (np.abs(beta_matrix) > 30).any(axis=1)
            iter[br] = maxit
            toFit[br] = False
    else:
        for i in range(maxit):
            iter[toFit] += 1
            if useWeights:
                w = weights[toFit] * \
                    (mu[toFit]/(1+(dispersions[toFit]*mu[toFit].T)).T)
                w_sqrt = np.sqrt(w)
            else:
                w = mu[toFit]/(1+(dispersions[toFit]*mu[toFit].T)).T
                w_sqrt = np.sqrt(w)
            z = np.log(mu[toFit]/sizeFactors) + \
                ((counts[toFit] - mu[toFit])/mu[toFit])
            beta = np.array([np.linalg.solve(np.matmul(designMatrix.T, np.multiply(
                designMatrix, w[row, None]))+ridge, np.matmul(designMatrix.T, (z[row]*w[row]))) for row in range(toFit.sum())])
            beta_matrix[toFit] = beta

            mu = np.maximum(
                sizeFactors * np.exp(np.matmul(designMatrix, beta_matrix.T)).T, minmu)

            deviance = np.zeros(toFit.sum())

            for sample in range(n_samples):
                if useWeights:
                    deviance += (-2 * weights[toFit, sample] * np.log(nbinom.pmf(
                        k=counts[toFit, sample], n=size[toFit], p=size[toFit]/(size[toFit]+mu[toFit, sample]))))
                else:
                    deviance += \
                        (-2 * np.log(nbinom.pmf(k=counts[toFit, sample],
                         n=size[toFit], p=size[toFit]/(size[toFit]+mu[toFit, sample]))))

            convergence = np.abs(
                deviance-deviances[toFit])/(np.abs(deviance) + 0.1)

            deviances[toFit] = deviance
            if i > 0:
                toFit[toFit] = (convergence >= betaTolerance)
                if all(toFit == False):
                    break
            br = (np.abs(beta_matrix) > 30).any(axis=1)
            iter[br] = maxit
            toFit[br] = False

    mu = np.maximum(
        sizeFactors * np.exp(np.matmul(designMatrix, beta_matrix.T)).T, minmu)
    if useWeights:
        w = weights * (mu/(1+(dispersions*mu.T)).T)
        w_sqrt = np.sqrt(w)
    else:
        w = mu/(1+(dispersions*mu.T)).T
        w_sqrt = np.sqrt(w)
    for row in range(n_rows):
        diagonal = np.zeros(n_samples, dtype=np.float64, order="C")
        dMws = np.multiply(designMatrix, w_sqrt[row][:, None])
        dMw = np.multiply(designMatrix, w[row][:, None])
        dMtwdMr_inv = np.linalg.inv(
            np.matmul(designMatrix.T, dMw)+ridge)
        # Verbose, but fast way to get diagonal of: matrix = dMws * dMtwdMr_inv * dMws.T
        for jp in range(n_samples):
            for idx1 in range(n_variables):
                for idx2 in range(n_variables):
                    diagonal[jp] = diagonal[jp] + \
                        (dMws[jp, idx1] * (dMws[jp, idx2]*dMtwdMr_inv[idx2, idx1]))

        diagonals[row] = diagonal
        # sigma = np.matmul(np.matmul(np.matmul(
        #     dMtwdMr_inv, designMatrix.T), dMw), dMtwdMr_inv)
        sigma = dMtwdMr_inv @ designMatrix.T @ dMw @ dMtwdMr_inv
        contrast_numerator[row] = np.matmul(contrast, beta_matrix[row])
        contrast_denominator[row] = np.sqrt(
            np.matmul(np.matmul(contrast, sigma), contrast))
        beta_var_matrix[row] = np.diag(sigma)

    return {
        "beta_matrix": beta_matrix,
        "beta_var_matrix": beta_var_matrix,
        "iter": iter,
        "diagonals": diagonals,
        "contrast_numerator": contrast_numerator,
        "contrast_denominator": contrast_denominator,
        "deviances": deviances
    }


cdef nbinomLogLike(
            counts,
            mu,
            dispersions,
            weights,
            useWeights
        ):
    if dispersions is None:
        return None
    if useWeights:
        size = 1/dispersions[:, None]
        return np.sum(weights * np.log(nbinom.pmf(k=counts, p=size/(size+mu), n=size)), axis=1)
    else:
        size = 1/dispersions[:, None]
        return np.sum(np.log(nbinom.pmf(k=counts, p=size/(size+mu), n=size)), axis=1)


cpdef fitNegativeBinomial(
        counts,
        normalizedCounts,
        sizeFactors,
        designMatrix,
        dispersions,  # np.ndarray[double, ndim=1, mode="c"]
        betaTolerance,  # 1e-8
        lambda_term,  # None
        maxit,  # 100
        useOptim,  # True
        forceOptim,  # False
        useQR,  # True
        weights,  # None
        useWeights,  # False
        minmu,  # 0.5
        ):

    nterms = designMatrix.shape[1]

    if lambda_term is None:
        lambda_term = np.full(nterms, 1E-6, order="C")  # np.tile(1E-6, nterms)

    if nterms == 1 and all(designMatrix == 1) and all(lambda_term <= 1E-6):
        betaConv = np.ones(len(counts), dtype=bool, order="C")
        betaIter = np.ones(len(counts), dtype=np.int32, order="C")
        if useWeights:
            betaMatrix = np.log2(
                np.sum((weights * normalizedCounts), axis=1) / np.sum(weights, axis=1)).reshape(-1, 1)
        else:
            betaMatrix = np.log2(
                np.mean(normalizedCounts, axis=1)).reshape(-1, 1)
        mu = sizeFactors * (2**betaMatrix)
        size = 1/dispersions[:, None]
        logLike_matrix = np.log(nbinom.pmf(k=counts, p=size/(size+mu), n=size))
        if useWeights:
            logLike = np.sum(weights * logLike_matrix, axis=1)
            w = weights * pow((pow(mu, -1)+dispersions[:, None]), -1)
        else:
            logLike = np.sum(logLike_matrix, axis=1)
            w = pow(pow(mu, -1)+dispersions[:, None], -1)
        sigma = pow(np.sum(w, axis=1), -1)
        betaSE = (np.log2(exp(1)) * np.sqrt(sigma))
        diagonals = w * sigma[:, None]

        return {"logLike": logLike, "betaConv": betaConv, "betaMatrix": betaMatrix, "betaSE": betaSE, "mu": mu, "betaIter": betaIter, "designMatrix": designMatrix, "nterms": 1, "diagonals": diagonals}

    if np.linalg.matrix_rank(designMatrix.values) == nterms:
        Q, R = np.linalg.qr(designMatrix.values)
        y = np.transpose(np.log(normalizedCounts + 0.1))
        beta_matrix = np.linalg.solve(R, (Q.T @ y)).T
    elif "Intercept" in designMatrix.columns:
        beta_matrix = np.zeros((len(counts), nterms), order="C")
        beta_matrix[:, 0] = np.log(np.mean(normalizedCounts, axis=1))
    else:
        beta_matrix = np.ones((len(counts), nterms), order="C")

    lambdaNatLogScale = lambda_term / pow(log(2), 2)

    betaResults = fitBeta(
        counts,  # counts
        designMatrix.values,  # designMatrix
        sizeFactors,  # sizeFactors
        dispersions,  # dispersions
        np.ascontiguousarray(beta_matrix),  # beta_matrix
        lambdaNatLogScale,  # lambda_term
        weights,  # weights
        useWeights,  # useWeights
        betaTolerance,  # betaTolerance
        maxit,  # maxit
        useQR,  # useQR
        minmu  # minmu
    )

    mu = sizeFactors * np.exp(designMatrix.values @
                              betaResults["beta_matrix"].T).T
    logLike = nbinomLogLike(counts, mu, dispersions, weights, useWeights)

    rowStable = (betaResults["beta_matrix"] != None).any(axis=1)
    rowVarPositive = (betaResults["beta_var_matrix"] > 0).all(axis=1)

    betaConv = betaResults["iter"] < maxit

    betaMatrix = np.log2(exp(1)) * betaResults["beta_matrix"]
    betaSE = np.log2(exp(1)) * \
        np.sqrt(np.maximum(betaResults["beta_var_matrix"], 0))

    if forceOptim:
        rowsForOptim = np.arange(len(betaConv))
    else:
        if useOptim:
            rowsForOptim = np.where(np.logical_not(betaConv) | np.logical_not(
                rowStable) | np.logical_not(rowVarPositive))[0]
        else:
            rowsForOptim = np.where(np.logical_not(
                rowStable) | np.logical_not(rowVarPositive))[0]

    if rowsForOptim.size > 0:
        optimResults = fitNegativeBinomialOptim(
                                            designMatrix,
                                            lambda_term,
                                            rowsForOptim,
                                            rowStable,
                                            sizeFactors,
                                            normalizedCounts,
                                            dispersions,
                                            weights,
                                            useWeights,
                                            betaMatrix,
                                            beta_matrix,
                                            betaSE,
                                            betaConv,
                                            mu,
                                            logLike,
                                            minmu
                                        )
        betaMatrix = optimResults["betaMatrix"]
        betaSE = optimResults["betaSE"]
        betaConv = optimResults["betaConv"]
        mu = optimResults["mu"]
        logLike = optimResults["logLike"]

    return {"logLike": logLike, "betaConv": betaConv, "betaMatrix": betaMatrix, "betaSE": betaSE, "mu": mu, "betaIter": betaResults["iter"], "designMatrix": designMatrix, "nterms": nterms, "diagonals": betaResults["diagonals"]}


def fitNegativeBinomialWithPrior(
            counts,
            normalizedCounts,
            sizeFactors,
            designMatrix,
            dispersionsResults,
            baseMean,  # baseMean
            betaTolerance,  # 1e-8
            lambda_term,  # None
            maxit,  # 100
            useOptim,  # True
            forceOptim,  # False
            useQR,  # True
            weights,  # None
            useWeights,  # False
            betaPriorVar,  # None
            mu,  # None
            MLE_betaMatrix,  # None
            minmu,  # 0.5
            betaPriorMethod,  # "weighted"
            upperQuantile,  # 0.05
        ):

    # dispersions is never going to be None
    dispersions = dispersionsResults["dispersion"].values

    if betaPriorVar is None or (dispersions is None and mu is None):
        fitWithOutPrior = fitNegativeBinomial(
                                        counts,
                                        normalizedCounts,
                                        sizeFactors,
                                        designMatrix,
                                        dispersions,
                                        betaTolerance,
                                        lambda_term,
                                        maxit,
                                        useOptim,
                                        forceOptim,
                                        useQR,
                                        weights,
                                        useWeights,
                                        minmu
                                    )
        designMatrix = fitWithOutPrior["designMatrix"]
        diagonals = fitWithOutPrior["diagonals"]
        # betaMatrix = fitWithOutPrior["betaMatrix"]
        mu = fitWithOutPrior["mu"]

        # MLE_betaMatrix = betaMatrix
        MLE_betaMatrix = fitWithOutPrior["betaMatrix"]
    if betaPriorVar is None:
        betaPriorVar = estimateBetaPriorVar(
            MLE_betaMatrix, designMatrix, normalizedCounts, dispersionsResults, baseMean, betaPriorMethod, upperQuantile)
    else:
        if len(betaPriorVar) != designMatrix.shape[1]:
            raise RuntimeError(
                f"betaPriorVar should have length {designMatrix.shape[1]}")
    if any(betaPriorVar == 0):
        print("beta prior variances are equal to zero for some variables")
    lambda_term = 1/betaPriorVar
    fit = fitNegativeBinomial(
                        counts,
                        normalizedCounts,
                        sizeFactors,
                        designMatrix,
                        dispersions,
                        betaTolerance,
                        lambda_term,
                        maxit,
                        useOptim,
                        forceOptim,
                        useQR,
                        weights,
                        useWeights,
                        minmu)

    return {"fit": fit, "diagonals": diagonals, "betaPriorVar": betaPriorVar, "mu": mu, "designMatrix": designMatrix, "MLE_betaMatrix": MLE_betaMatrix}


cdef fitNegativeBinomialOptim(
            designMatrix,
            lambda_term,
            rowsForOptim,
            rowStable,
            sizeFactors,
            normalizedCounts,
            dispersions,
            weights,  # None
            bint useWeights,  # False
            betaMatrix,
            beta_matrix,
            betaSE,
            betaConv,
            mu,
            logLike,
            double minmu  # 0.5
        ):

    cdef:
        # Py_ssize_t row = 0
        np.ndarray[double, ndim = 1, mode = "c"] lambdaNatLogScale = lambda_term / pow(log(2), 2)
        int large = 30
    for row in rowsForOptim:
        if rowStable[row] and all(np.abs(betaMatrix[row]) < large):
            betaRow = betaMatrix[row]
        else:
            betaRow = beta_matrix[row]
        counts_row = normalizedCounts[row]  # k
        dispersion = dispersions[row]
        if useWeights:
            weight = weights[row]

        def objectiveFn(p):
            mu_row = sizeFactors * (2**(designMatrix.values @ p))
            size = 1/dispersion
            logLikeVector = np.log(nbinom.pmf(
                    k=counts_row, n=size, p=size/(size+mu_row)))
            if useWeights:
                logLike = weight.sum() * logLikeVector
            else:
                logLike = logLikeVector.sum()
            logPrior = np.log(norm.pdf(p, loc=0, scale=1/lambda_term)).sum()
            negLogPost = -1 * (logLike + logPrior)
            if np.isfinite(negLogPost):
                return negLogPost
            else:
                return pow(10, 300)

        o = minimize(objectiveFn, betaRow, method="L-BFGS-B",
                     bounds=[(-large, large) for _ in betaRow])
        if len(lambdaNatLogScale) > 1:
            ridge = np.diag(lambdaNatLogScale)
        else:
            ridge = lambdaNatLogScale
        if o.success:
            betaConv[row] = True
        betaMatrix[row] = o.x
        mu_row = sizeFactors * (2**(designMatrix.values @ o.x))
        mu[row] = mu_row
        mu_row = np.maximum(mu_row, minmu)
        if useWeights:
            w = np.diag(weight * pow((pow(mu_row, -1) + dispersion), -1))
        else:
            w = np.diag(pow((pow(mu_row, -1)+dispersion), -1))
        xtwx = designMatrix.values.T @ w @ designMatrix.values
        xtwxRidgeInv = np.linalg.inv(xtwx + ridge)
        sigma = xtwxRidgeInv @ xtwx @ xtwxRidgeInv
        betaSE[row] = np.log2(exp(1)) * np.sqrt(np.maximum(np.diag(sigma), 0))
        size = 1/dispersion
        logLikeVector = np.log(nbinom.pmf(
                k=counts_row, n=size, p=size/(size+mu_row)))
        if useWeights:
            logLike[row] = weight.sum() * logLikeVector
        else:
            logLike[row] = logLikeVector.sum()

    return {"betaMatrix": betaMatrix, "betaSE": betaSE, "betaConv": betaConv, "mu": mu, "logLike": logLike}


cdef estimateBetaPriorVar(MLE_betaMatrix, designMatrix, normalizedCounts, dispersionsResults, baseMean, betaPriorMethod, upperQuantile):
    if dispersionsResults is None:
        raise RuntimeError("run the analysis first")
    if betaPriorMethod is None:
        betaPriorMethod = "weighted"
    if upperQuantile is None:
        upperQuantile = 0.05

    dispFit = dispersionsResults["dispFit"].values

    weights = 1/(1/baseMean + dispFit)

    if MLE_betaMatrix.shape[0] > 1:
        def col_betaPriorVar(x):
            useFinite = np.abs(x) < 10
            if np.sum(np.abs(x) < 10) == 0:
                return 1E6
            else:
                if betaPriorMethod == "quantile":
                    return matchUpperQuantileForVariance(x[useFinite], upperQuantile, weights)
                elif betaPriorMethod == "weighted":
                    return matchUpperQuantileForVariance(x[useFinite], upperQuantile, weights[useFinite])
        betaPriorVar = np.apply_along_axis(
            func1d=lambda x: col_betaPriorVar(x), axis=0, arr=MLE_betaMatrix)
    else:
        betaPriorVar = pow(MLE_betaMatrix[0], 2)

    if "Intercept" in designMatrix.columns:
        betaPriorVar[np.where(designMatrix.columns == "Intercept")[
                              0]] = 1E6  # intercept set to wide prior

    return betaPriorVar


cpdef matchUpperQuantileForVariance(
            np.ndarray[np.float64_t, ndim=1, mode="c"] x,
            double upperQuantile,  # 0.05
            np.ndarray[np.float64_t, ndim=1, mode="c"]  weights  # None
        ):
    cdef:
        double quantile
        np.ndarray[np.int32_t, ndim = 1, mode = "c"] sorter
        double sum_weights
        double order
        double low
        double high
        double[::1] all_quantiles
    if weights is None:
        quantile = np.quantile(np.abs(x), 1-upperQuantile)
    else:
        x = np.abs(x[weights != 0])
        weights = weights[weights != 0]
        sorter = np.argsort(x)
        x = x[sorter]
        weights = weights[sorter]
        # to make weights sum to length(x) after deletion of NAs
        weights = weights * len(x)/np.sum(weights)
        sum_weights = np.sum(weights)
        order = 1 + (sum_weights - 1) * (1-upperQuantile)
        low = max(int(order), 1)
        high = min(low+1, sum_weights)
        order %= 1
        all_quantiles = interp1d(np.cumsum(
            weights), x, kind="next", assume_sorted=True, fill_value="extrapolate")([low, high])
        quantile = (1-order)*all_quantiles[0] + \
            order*all_quantiles[len(all_quantiles)-1]

    return pow((quantile / norm.ppf(1 - upperQuantile/2)), 2)
