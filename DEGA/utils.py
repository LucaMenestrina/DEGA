import pandas as pd
import numpy as np

from scipy.stats import trim_mean


def buildResultsDataFrame(results, index=None, designMatrix=None, NArows=None):
    """
        Gets a dictionary of results and returns a matrix
        with the dictionary's values (lists/arrays) as columns
        ("explodes" them using designMatrix columns when needed)
    """
    def splitMatrixColsInDict(d, designMatrix=None):
        if not designMatrix is None:
            designMatrixColumns = designMatrix.columns
        for key, value in d.items():
            if not key is None:
                if (isinstance(value, list) or
                        (
                            (isinstance(value, np.ndarray)
                             or isinstance(value, pd.DataFrame))

                            and value.ndim == 1
                        )
                        or isinstance(value, pd.Series)):
                    yield key, value
                elif ((isinstance(value, np.ndarray) or isinstance(value, pd.DataFrame))
                        and value.ndim > 1):
                    if designMatrix is None:
                        raise RuntimeError(
                            "Results dict has values which are numpy ndarrays with ndim > 1, 'designMatrix' is required")
                    for k, v in zip(
                                        (f"{key} ({dmCol})" for dmCol in designMatrixColumns),
                                        (arr.flatten()
                                         for arr in np.hsplit(value, value.shape[1]))
                                    ):
                        yield k, v

    results = dict(splitMatrixColsInDict(
        results, designMatrix=designMatrix))
    if NArows is None:
        return pd.DataFrame(results, columns=list(results.keys()), index=index)
    else:
        if not isinstance(NArows, np.ndarray):
            NArows = np.array(NArows)
        A = np.empty((len(NArows), len(results)), dtype="object")
        A[NArows] = np.NaN
        A[~NArows] = np.array(list(results.values()), dtype="object").T
        return pd.DataFrame(A,
                            columns=list(results.keys()),
                            index=index).astype(dict(
                                                    zip(list(results.keys()),
                                                        [val.dtype.type for val in results.values()])
                                                    )
                                                )


def getBaseMeansAndVariances(normalizedCounts, weights=None):
    normCounts = normalizedCounts.copy()
    if not weights is None:
        normCounts = normCounts * weights

    allZero = normCounts.sum(axis=1) == 0
    baseMean = normCounts.mean(axis=1)
    baseVar = normCounts.var(axis=1, ddof=1)

    return buildResultsDataFrame(
        {"baseMean": baseMean, "baseVar": baseVar, "allZero": allZero})


def median_absolute_deviation(data, constant=1.4826):
    """
        MAD
        default constant ensures consistency
        (check https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/mad)
    """
    return 1.4826 * np.median(
        np.absolute(data - np.median(data))
        )


def sufficientReplicates(designMatrix, minReplicates):
    _, idx, counts = np.unique(
        designMatrix.values, return_inverse=True, return_counts=True, axis=0)
    return (counts >= minReplicates)[idx]


def trimmedVariance(normalizedCounts):
    tmean = trim_mean(normalizedCounts, 1/8, axis=1)
    sqerror = (normalizedCounts-tmean.reshape((-1, 1)))**2
    # scale due to trimming of large squares
    var = 1.51 * trim_mean(sqerror, 1/8, axis=1)
    return var


def trimmedSampleVariance(normalizedCounts, sample_level):
    def trimfn(x): return 1/3 if 0 < x <= 3.5 else 1 / \
               4 if 3.5 < x <= 23.5 else 1/8 if 23.5 < x else None
    valid_sample_levels, valid_sample_idx, valid_sample_counts = np.unique(
        sample_level, return_inverse=True, return_counts=True)
    sample_means = np.array([trim_mean(normalizedCounts[:, sample_level == valid_sample_levels[i]],
                                       trimfn(n), axis=1) for i, n in enumerate(valid_sample_counts)]).T
    sqerror = (normalizedCounts - sample_means[:, valid_sample_idx])**2

    def scalefn(
        x): return 2.04 if 0 < x <= 3.5 else 1.86 if 3.5 < x <= 23.5 else 1.51 if 23.5 < x else None
    var = np.array([scalefn(n) * trim_mean(sqerror[:, sample_level
                   == valid_sample_levels[i]], trimfn(n), axis=1) for i, n in enumerate(valid_sample_counts)]).T
    max_var = np.max(var, axis=1)
    return max_var
