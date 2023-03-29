import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import gmean

from DEGA.utils import getBaseMeansAndVariances, sufficientReplicates
from DEGA.dispersions import (
    estimateDispersionsGeneEst,
    estimateDispersionsFit,
    estimateDispersionsMAP,
    estimateDispersions,
)
from DEGA.statisticalTests import WaldTest, LRTTest, adjustForOutliers
from DEGA.refit import refitWithoutOutliers
from DEGA.countsTranformers import shiftedLog, vst, rlog
from DEGA.plots import (
    plotDispersions,
    plotMA,
    plotVolcano,
    plotSparsity,
    plotSamplesCorrelation,
    plotSamplesGenesCorrelation,
    plotPCAExplainedVariance,
    plotPCAComponentsDensity,
    plotPCA,
    plotIndependentFiltering,
    plotCooks,
    plotCoexpressionNetwork,
)

from DEGA.logger import log


class dataset():
    """
    DEGA Dataset

    The primary DEGA class.
    Stores info about inputs (gene counts, phenotype data)
    as well as outputs (log2 Fold Change, adjusted p-values, etc.).

    Parameters
    ----------
    geneCounts : pandas.DataFrame
        Matrix of the read counts (non-negative integers).
        Columns of geneCounts correspond to rows of phenotypeData.

    phenotypeData : pandas.DataFrame
        Matrix with at least a column describing the tested phenotypes.
        Rows of phenotypeData correspond to columns of geneCounts.

    designFormula : str
        Expresses how the counts for each gene
        depend on the variables in phenotypeData.

    weights : numpy.ndarray(dtype=float, ndim=2), optional
        2D array of weights.

    useWeights : bool, default=False
        Whether to use provided weights.

    Notes
    -----
    geneCounts columns must be provided in the same order of phenotypeData index.

    Example
    -------
    >>> geneCounts = pd.read_csv("path/to/geneCountsFile.csv", index_col=0)
    >>> phenotypeData = pd.read_csv("path/to/phenotypeDataFile.csv", index_col=0)
    >>> geneCounts = geneCounts[phenotypeData.index]
    >>> DEGA.dataset(geneCounts=geneCounts, phenotypeData=phenotypeData, designFormula="designFormula")

    """

    def __init__(self, geneCounts, phenotypeData, designFormula=None, weights=None, useWeights=False) -> None:
        """
        Initialize self.

        Loads input data. Computes size factors and normalizes counts.
        """
        log.info("Loading Data")
        self.__phenotypeData = phenotypeData.copy()
        self.__geneCounts = geneCounts.copy()

        self.__genes = np.array(self.__geneCounts.index)
        self.__counts = self.__geneCounts.copy()

        # Compute sizeFactors
        _pseudo_ref = gmean(self.__counts.values, axis=1).reshape(-1, 1)
        self.__sizeFactors = np.median(
            self.__counts.values[np.where(_pseudo_ref != 0)[0]]
            / _pseudo_ref[np.where(_pseudo_ref != 0)[0]],
            axis=0,
        )
        self.__normalizedCounts = self.__counts / self.__sizeFactors
        if not designFormula is None:
            self.__designFormula = designFormula
            self.__designMatrix = dmatrix(
                self.__designFormula, data=self.__phenotypeData, return_type="dataframe")
            self.__designMatrix.columns = np.array(self.__designMatrix.design_info.term_names)[
                                                   [coding[0].num_columns > 0 for coding in self.__designMatrix.design_info.term_codings.values()]]
        else:
            log.warning(
                "'designFormula' is not specified, using formula '~1' (just intercept)")
            self.__designFormula = "1"
            self.__designMatrix = dmatrix("1", data=self.__phenotypeData,
                                          return_type="dataframe")
            self.__designMatrix.columns = np.array(self.__designMatrix.design_info.term_names)[
                                                   [coding[0].num_columns > 0 for coding in self.__designMatrix.design_info.term_codings.values()]]

        self.__weights = weights
        self.__useWeights = useWeights

        self.__meanVarZero = getBaseMeansAndVariances(
            self.__normalizedCounts, self.__weights)

        self.__allZero = self.__meanVarZero["allZero"].copy().values
        self.__baseMean = self.__meanVarZero["baseMean"].values
        self.__baseVar = self.__meanVarZero["baseVar"].values

        # only continue on the rows with non-zero row mean
        self.__genes = self.__genes[~self.__allZero]
        self.__counts = self.__counts.loc[~self.__allZero, :]
        self.__normalizedCounts = self.__normalizedCounts.loc[~self.__allZero, :]
        self.__weights = self.__weights[~self.__allZero,
                                        :] if not self.__weights is None else None

        self.__meanVarZero = self.__meanVarZero.loc[~self.__allZero, :]
        self.__baseMean = self.__baseMean[~self.__allZero]
        self.__baseVar = self.__baseVar[~self.__allZero]
        self.__allZero = self.__allZero[~self.__allZero]

        self.summary

    def analyse(self, counts=None, normalizedCounts=None, phenotypeData=None,
                sizeFactors=None, meanVarZero=None, designMatrix=None, dispPriorVar=None,
                weights=None, useWeights=False, weightThreshold=1e-2, minDisp=1e-8,
                kappa0=1.0, dispTolerance=1e-6, betaTolerance=1e-6, niter=1,
                linearMu=None, alphaInit=None, fitType="parametric", outlierSD=2,
                test="Wald", lfcThreshold=0, shrink=True, reducedDesignMatrix=None,
                useCR=True, minmu=0.5, maxit=100, useOptim=True, forceOptim=False,
                useT=False, useQR=True, alpha=0.05, pAdjustMethod="fdr_bh",
                altHypothesis="greaterAbs", dof=None, betaPriorVar=None,
                betaPriorMethod="weighted", upperQuantile=0.05, minReplicatesForReplace=7,
                cooksCutoff=None, compute_d2log_posterior=False,
                ) -> None:
        """
        Main function for running the differential expression analysis.

        Parameters
        ----------
        counts : pandas.DataFrame, optional
            Matrix of the read counts (non-negative integers).
            By default, the one provided as `geneCounts`
            when loading the dataset is used.

        normalizedCounts : pandas.DataFrame, optional
            Matrix of normalized counts.
            By default, the ones normalized
            dividing the `counts` by the `sizeFactors`
            when loading the dataset is used.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        sizeFactors : numpy.ndarray(dtype=float, ndim=1), optional
            Multiplicative normalizing factors for each sample.

        meanVarZero : pandas.DataFrame, optional
            DataFrame with row means and variances of normalized counts.
            Has a boolean column specifying if the row sums are zero.

        designMatrix : pandas.DataFrame, optional
            A custom matrix.
            By default, the one built from the `designFormula`
            when loading the dataset is used.

        dispPriorVar : float, optional
            The variance of the normal prior on the log dispersions.
            By default, it is calculated as the difference between
            the mean squared residuals of gene-wise estimates to the
            fitted dispersion and the expected sampling variance
            of the log dispersion.

        weights : numpy.ndarray(dtype=float, ndim=2), optional
            2D array of weights

        useWeights : bool, default=False
            Whether to use provided weights.

        weightThreshold : float, default=1e-2
            Threshold for subsetting the design matrix and GLM weights
            for calculating the Cox-Reid correction.

        minDisp : float, default=1e-8
            Small value for the minimum dispersion, to allow for calculations in log scale,
            one order of magnitude above this value is used
            as a test for inclusion in mean-dispersion fitting.

        kappa0 : float, default=1.0
            Control parameter used in setting the initial proposal
            in backtracking search, higher `kappa0` results in larger steps.

        dispTolerance : float, default=1e-6
            Control parameter to test for convergence of log dispersion,
            stop when increase in log posterior is less than `dispTolerance`.

        betaTolerance : float, default=1e-6
            Control parameter to test for convergence of deviances.

        niter : int, default=1
            Number of times to iterate between estimation of means and
            estimation of dispersion.

        linearMu : bool, optional
            Estimate the expected counts matrix using a linear model.
            By default, a linear model is used if the number of groups
            defined by the model matrix is equal to the number of columns
            of the design matrix.

        alphaInit : numpy.ndarray(dtype=float, ndim=1), optional
            Initial guess for the dispersion estimate, alpha.

        fitType : {"parametric", "local", "mean", "kernel"}, default="parametric"
            Either "parametric", "local", "mean", or "kernel"
            for the type of fitting of dispersions to the mean intensity.

        test : {"Wald", "LRT"}
            Either "Wald" or "LRT", for the type of significance test to be used:
            Wald significance tests or the likelihood ratio test
            on the difference in deviance between a full and reduced design formula.

        lfcThreshold : float, default=0
            A non-negative value which specifies a threshold for the log2 fold change.

        shrink : bool, default=True
            Whether to shrink the log2 fold changes and standard error.

        reducedDesignMatrix : pandas.DataFrame, optional
            Only if `test `is "LRT".
            A reduced design matrix to compare against i.e.,
            the full design matrix with the term(s) of interest removed.
            Alternatively, it can be a design matrix constructed by the user.

        useCR : bool, default=True
            Whether to use Cox-Reid correction.

        minmu : float, default=0.5
            Lower bound on the estimated count for fitting gene-wise dispersion
            and for statistical testing.

        maxit : int, default=100
            Maximum number of iterations to allow for convergence.

        useOptim : bool, default=True
            Whether to use the native optim function on rows which do not
            converge within `maxit`.

        forceOptim : bool, default=False
            Whether to use optim on all rows.

        useT : bool, default=False
            Whether to use a t-distribution as a null distribution,
            for significance testing of the Wald statistics.
            If False, a standard normal null distribution is used.

        useQR : bool, default=True
            Whether to use the QR decomposition on the design
            matrix while fitting the GLM.

        alpha : float, default=0.05
            The significance cutoff used for optimizing the independent
            filtering.

        pAdjustMethod : str, default="fdr_bh"
            The method to use for adjusting p-values for multiple tests.
            Check `method` parameter of the function `multipletests`
            from from `statsmodels.stats.multitest`.

        altHypothesis : {"greaterAbs", "lessAbs", "greater", "less"}, default="greaterAbs"
            The alternative hypothesis, i.e. those values of log2 fold change which the user is interested in finding.
            The complement of this set of values is the null hypothesis which will be tested.
            The possible values represent the following alternate hypotheses:
             - "greaterAbs": $|LFC| > lfcThreshold$ and p-values are two-tailed.
             - "lessAbs": $|LFC| < lfcThreshold$ and p-values are the maximum of the upper and lower tests.
             - "greater": $LFC > lfcThreshold$.
             - "less": $LFC < -lfcThreshold$.

        dof : int, optional
            The degrees of freedom for the t-distribution.
            By default, the degrees of freedom will be set
            by the number of samples minus the number of columns of the design
            matrix used for dispersion estimation.
            If `weights` are used, then the sum of the weights is used in lieu
            of the number of samples.

        betaPriorVar : numpy.ndarray(dtype=float, ndim=1), optional
            A vector with length equal to the number of design parameters
            including the intercept.
            `betaPriorVar` gives the variance of the prior on the sample betas
            on the log2 scale.
            By default, it is estimated from the data.

        betaPriorMethod : {"weighted", "quantile"}, default="weighted"
            The method for calculating the beta prior variance,
            either "weighted" or "quantile":
             - "quantile" matches a normal distribution using the upper quantile of the finite MLE betas.
             - "weighted" matches a normal distribution using the upper quantile,
            but weighting by the variance of the MLE betas.

        upperQuantile : float, default=0.05
            The upper quantile to be used for the "quantile" or "weighted"
            method of beta prior variance estimation.
            See `betaPriorMethod`.

        minReplicatesForReplace : int, default=7
            The minimum number of replicates required in order to refit for outliers.
            If there are samples with so many replicates, the model will
            be refit after replacing these outliers (flagged by Cook's distance).
            Set to ``np.Inf`` in order to never replace outliers.

        cooksCutoff : float, optional
            The threshold for defining an outlier to be replaced.
            By default, the 0.99 quantile of the F(p, m - p) distribution,
            where p is the number of parameters and m is the number of samples.

        compute_d2log_posterior : bool, default=False
            Whether to compute the second derivative of the log posterior
            with respect to the log of the dispersion parameter alpha.
            It will not affect the analysis, normal users should not change this parameter.

        Example
        -------
        >>> dega.analyse()

        """
        if counts is None:
            counts = self.counts
        if normalizedCounts is None:
            normalizedCounts = self.normalizedCounts
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if sizeFactors is None:
            sizeFactors = self.sizeFactors
        if meanVarZero is None:
            meanVarZero = self.meanVarZero
        if designMatrix is None:
            designMatrix = self.designMatrix
        self.__test = test
        self.__alpha = alpha
        self.__lfcThreshold = lfcThreshold
        self.__withShrinkage = shrink
        log.info("Estimating Dispersions")
        self.__dispersionsResults = self._estimateDispersions(counts, normalizedCounts, sizeFactors,
                                                              self.meanVarZero, designMatrix, dispPriorVar,
                                                              weights, useWeights, weightThreshold,
                                                              minDisp, kappa0, dispTolerance,
                                                              betaTolerance, niter, linearMu,
                                                              alphaInit, fitType, maxit, useCR,
                                                              minmu, outlierSD, compute_d2log_posterior)
        log.info("Hypothesis Testing")
        self.__testResults = self._testSignificance(counts, normalizedCounts, phenotypeData,
                                                    sizeFactors, self._dispersionsResults(),
                                                    self.meanVarZero, designMatrix, test,
                                                    reducedDesignMatrix, betaTolerance,
                                                    weights, useWeights, minmu, maxit,
                                                    useOptim, forceOptim, useT, useQR, alpha,
                                                    pAdjustMethod, dof, shrink, betaPriorVar,
                                                    betaPriorMethod, upperQuantile)

        if sufficientReplicates(designMatrix, minReplicatesForReplace).any():
            log.info("Refitting Without Outliers")
            (
                self.__meanVarZero,
                self.__dispersionsResults,
                self.__testResults,
                self.__replace,
                self.__replacedOutliersCounts
            ) = self.__refitWithoutOutliers(counts, normalizedCounts, self.phenotypeData,
                                            self.__cooks, sizeFactors, test, self.meanVarZero,
                                            designMatrix, reducedDesignMatrix, self._dispersionsResults(),
                                            self._dispFitFunc, self._testResults(
                                                showIntercept=True),
                                            dispPriorVar, weights, useWeights, weightThreshold, minDisp,
                                            kappa0, dispTolerance, betaTolerance, niter, linearMu, alphaInit,
                                            fitType, maxit, useOptim, forceOptim, useT, useQR, alpha,
                                            pAdjustMethod, dof, shrink, betaPriorVar, betaPriorMethod,
                                            upperQuantile, useCR, minmu, outlierSD, compute_d2log_posterior,
                                            minReplicatesForReplace, cooksCutoff)  # shrink stands for betaPrior
            self.__allZero = self.__meanVarZero["allZero"].values
            self.__baseMean = self.__meanVarZero["baseMean"].values
            self.__baseVar = self.__meanVarZero["baseVar"].values
        else:
            self.__replace = np.full(counts.shape[0], np.NaN)
            self.__replacedOutliersCounts = self.__counts

        log.info("Adjusting for Outliers")
        self.__testResults = adjustForOutliers(counts, phenotypeData, test, self.__testResults,
                                               designMatrix, self._betaPriorVar, self.baseMean,
                                               self.__cooks, self.__replace, cooksCutoff, alpha,
                                               lfcThreshold, useT, pAdjustMethod, altHypothesis)

        LFCColumnName = [col for col in self.results.columns if (
            col.startswith("log2 fold change") and "Intercept" not in col)][0]
        PvalueColumnName = [col for col in self.results.columns if (
            col.startswith("adjusted p-value") and "Intercept" not in col)][0]
        self.__upregulatedGenes = self.genes[(self.results[LFCColumnName] > lfcThreshold) & (
            self.results[PvalueColumnName] < alpha)]
        self.__downregulatedGenes = self.genes[(
            self.results[LFCColumnName] < -lfcThreshold) & (self.results[PvalueColumnName] < alpha)]

        log.info("Done!")

    def _estimateDispersions(self, counts, normalizedCounts, sizeFactors, meanVarZero,
                             designMatrix, dispPriorVar=None, weights=None, useWeights=False,
                             weightThreshold=1e-2, minDisp=1e-8, kappa0=1, dispTolerance=1e-6,
                             betaTolerance=1e-6, niter=1, linearMu=None, alphaInit=None,
                             fitType="parametric", maxit=100, useCR=True, minmu=0.5,
                             outlierSD=2, compute_d2log_posterior=False):
        (
            self.__dispersionsResults,
            self.__dispFitFunc,
            self.__fitParams,
            self.__mu
        ) = estimateDispersions(counts, normalizedCounts, sizeFactors, meanVarZero,
                                designMatrix, dispPriorVar, weights, useWeights,
                                weightThreshold, minDisp, kappa0, dispTolerance,
                                betaTolerance, niter, linearMu, alphaInit, fitType,
                                maxit, useCR, minmu, outlierSD, compute_d2log_posterior)
        self.__fitType = self._fitParams["fitType"]
        return self._dispersionsResults()

    def __estimateDispersionsGeneEst(self, genes, counts, normalizedCounts, sizeFactors,
                                     baseMean, baseVar, designMatrix, weights=None,
                                     useWeights=False, weightThreshold=1e-2, minDisp=1e-8,
                                     kappa0=1, dispTolerance=1e-6, betaTolerance=1e-8,
                                     maxit=100, useCR=True, niter=1, linearMu=None,
                                     minmu=0.5, alphaInit=None, compute_d2log_posterior=False):
        self.__dispGeneEstResults, self.__mu = estimateDispersionsGeneEst(genes, np.ascontiguousarray(counts),
                                                                          np.ascontiguousarray(
                                                                              normalizedCounts),
                                                                          sizeFactors, baseMean, baseVar, designMatrix,
                                                                          weights, useWeights, weightThreshold, minDisp,
                                                                          kappa0, dispTolerance, betaTolerance, maxit,
                                                                          useCR, niter, linearMu, minmu, alphaInit,
                                                                          compute_d2log_posterior)
        return self._dispGeneEstResults

    def __estimateDispersionsFit(self, genes, dispGeneEsts, baseMean, designMatrix, fitType="parametric", minDisp=1e-8):
        self.__dispFitResults, self.__dispFitFunc, self.__fitParams = estimateDispersionsFit(
            genes, dispGeneEsts, baseMean, designMatrix, fitType, minDisp)
        return self._dispFitResults

    def __estimateDispersionsMAP(self, genes, counts, dispGeneEsts, dispFit, mu,
                                 designMatrix, dispPriorVar=None, weights=None, useWeights=False,
                                 weightThreshold=1e-2, outlierSD=2, minDisp=1e-8, kappa0=1.0,
                                 dispTolerance=1e-6, maxit=100, useCR=True, compute_d2log_posterior=False):
        self.__dispMAPResults = estimateDispersionsMAP(genes, counts, dispGeneEsts, dispFit, mu,
                                                       designMatrix, dispPriorVar, weights, useWeights,
                                                       weightThreshold, outlierSD, minDisp, kappa0,
                                                       dispTolerance, maxit, useCR, compute_d2log_posterior)
        return self._dispMAPResults

    def _testSignificance(self, counts, normalizedCounts, phenotypeData, sizeFactors,
                          dispersionsResults, meanVarZero, designMatrix, test="Wald",
                          reducedDesignMatrix=None, betaTolerance=1e-8, weights=None,
                          useWeights=False, minmu=0.5, maxit=100, useOptim=True,
                          forceOptim=False, useT=False, useQR=True, alpha=0.05,
                          pAdjustMethod="fdr_bh", dof=None, betaPrior=False,
                          betaPriorVar=None, betaPriorMethod="weighted", upperQuantile=0.05):
        if test == "Wald":
            (
                self.__testResults,
                self.__cooks,
                self.__betaPriorVar
            ) = WaldTest(np.ascontiguousarray(counts), np.ascontiguousarray(normalizedCounts),
                         sizeFactors, dispersionsResults, meanVarZero, designMatrix,
                         betaTolerance, weights, useWeights, minmu, maxit, useOptim,
                         forceOptim, useT, useQR, dof, alpha, pAdjustMethod, betaPrior,
                         betaPriorVar, betaPriorMethod, upperQuantile)
        elif test == "LRT":
            if reducedDesignMatrix is None:
                log.warning(
                    "Reduced design matrix not provided. Using formula '~1'")
                self.__reducedDesignMatrix = dmatrix(
                    "1", data=phenotypeData, return_type="dataframe")
            (
                self.__testResults,
                self.__cooks,
                self.__betaPriorVar
            ) = LRTTest(np.ascontiguousarray(counts), np.ascontiguousarray(normalizedCounts),
                        sizeFactors, dispersionsResults, meanVarZero, designMatrix,
                        self.__reducedDesignMatrix, betaTolerance, weights, useWeights,
                        minmu, maxit, useOptim, forceOptim, useT, useQR, alpha, pAdjustMethod)
        else:
            raise RuntimeError(
                f"{test} test not implemented, choose one of 'Wald' and 'LRT'")
        return self.__testResults

    def __refitWithoutOutliers(self, counts, normalizedCounts, phenotypeData, cooks,
                               sizeFactors, test, meanVarZero, fullDesignMatrix,
                               reducedDesignMatrix, dispersionsResults, dispFitFunc,
                               testResults, dispPriorVar=None, weights=None,
                               useWeights=False, weightThreshold=1e-2, minDisp=1e-8,
                               kappa0=1, dispTolerance=1e-6, betaTolerance=1e-8,
                               niter=1, linearMu=None, alphaInit=None, fitType="parametric",
                               maxit=100, useOptim=True, forceOptim=False, useT=False,
                               useQR=True, alpha=0.05, pAdjustMethod="fdr_bh", dof=None,
                               betaPrior=False, betaPriorVar=None, betaPriorMethod="weighted",
                               upperQuantile=0.05, useCR=True, minmu=0.5, outlierSD=2,
                               compute_d2log_posterior=False, minReplicatesForReplace=7, cooksCutoff=None):
        # testResults must be provided with data about intercept
        return refitWithoutOutliers(counts, normalizedCounts, phenotypeData, cooks,
                                    sizeFactors, test, meanVarZero, fullDesignMatrix,
                                    reducedDesignMatrix, dispersionsResults, dispFitFunc,
                                    testResults, dispPriorVar, weights, useWeights,
                                    weightThreshold, minDisp, kappa0, dispTolerance,
                                    betaTolerance, niter, linearMu, alphaInit, fitType,
                                    maxit, useOptim, forceOptim, useT, useQR, alpha,
                                    pAdjustMethod, dof, betaPrior, betaPriorVar,
                                    betaPriorMethod, upperQuantile, useCR, minmu,
                                    outlierSD, compute_d2log_posterior,
                                    minReplicatesForReplace, cooksCutoff)

    def plotDispersions(self, dispersionsResults=None, baseMean=None, geneColor="black", fitColor="red",
                        finalColor="dodgerblue", s=2, linewidth=1, legend=True, path=None):
        """
        Technical plot for displaying the dispersion estimates.

        Parameters
        ----------
        dispersionsResults : pandas.DataFrame, optional
            Results of the dispersions estimations.

        baseMean : numpy.ndarray(dtype=float, ndim=1), optional
            Row means of normalized counts.

        geneColor : str, defaul="black"
            Color for gene-wise dispersion estimates

        fitColor : str, default="red"
            Color for the fitted dispersions.

        finalColor : str, default="dodgerblue"
            Color for the final dispersion estimates.

        s : float, default=2
            Marker size.

        linewidth : float, default=1,
            Linewidth of marker edges for outliers.

        legend : bool, default=True
            Whether to plot the legend.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if dispersionsResults is None:
            dispersionsResults = self._dispersionsResults()
        if baseMean is None:
            baseMean = self.baseMean
        return plotDispersions(dispersionsResults, baseMean, geneColor, fitColor,
                               finalColor, s, linewidth, legend, path)

    def plotMA(self, testResults=None, baseMean=None, pValueThreshold=None, lfcThreshold=None,
               boundedY=True, legend=True, highlightGenes=None, labels=True, path=None):
        """
        MA-plot showing the log2 fold changes over the mean of normalized counts.

        Parameters
        ----------
        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        baseMean : numpy.ndarray(dtype=float, ndim=1), optional
            Row means of normalized counts.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        boundedY : bool, default=True
            Whether to set limits for the Y axis.

        legend : bool, default=True
            Whether to plot the legend.

        highlightGenes : list, optional
            Which genes to highlight with a green circle.

        labels : bool, default=True
            Whether to plot the labels for highlighted genes.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if testResults is None:
            testResults = self._testResults()
        if baseMean is None:
            baseMean = self.baseMean
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        return plotMA(testResults, baseMean, pValueThreshold,
                      lfcThreshold, boundedY, legend, highlightGenes, labels, path)

    def plotVolcano(self, testResults=None, pValueThreshold=None, lfcThreshold=None, boundedY=True,
                    labels=True, labelsQuantile=0.005, legend=True, highlightGenes=None, highlightedGenesLabels=True, path=None):
        """
        Volcano plot.

        Parameters
        ----------
        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        boundedY : bool, default=True
            Whether to set limits for the Y axis.

        labels : bool, default=True
            Whether to plot the labels for relevant genes
            (adjusted p-value < `pValueThreshold` and log2 fold change > `lfcThreshold`).

        labelsQuantile : float, default=0.005
            A non-negative value which specifies the quantile of gene labels to be displayed
            (starting from the furthest from `pValueThreshold` and `lfcThreshold`).
            If set to 1 or False: display all labels.

        legend : bool, default=True
            Whether to plot the legend.

        highlightGenes : list, optional
            Which genes to highlight with a green circle.

        highlightedGenesLabels : bool, default=True
            Whether to plot the labels for highlighted genes

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        return plotVolcano(testResults, pValueThreshold, lfcThreshold, boundedY,
                           labels, labelsQuantile, legend, highlightGenes, highlightedGenesLabels, path)

    def plotSparsity(self, normalizedCounts=None, path=None):
        """
        Display sparsity of normalized counts.

        Parameters
        ----------
        normalizedCounts : pandas.DataFrame, optional
            Matrix of normalized counts.
            By default, the ones normalized
            dividing the `counts` by the `sizeFactors`
            when loading the dataset is used.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if normalizedCounts is None:
            normalizedCounts = self.normalizedCounts

        return plotSparsity(normalizedCounts, path)

    def plotSamplesCorrelation(self, transformedCounts=None, testResults=None, phenotypeData=None,
                               phenotypeColumn=None, pValueThreshold=None, lfcThreshold=None,
                               method="spearman", clustering=True, path=None):
        """
        Plot a matrix of the correlations between samples.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        phenotypeColumn : str
            Name of the column of the `phenotypeData` DataFrame
            which contains the data about the phenotype of interest.
            If not provided `DEGA` tries to retrive it from the design matrix,
            if it fails it raises an error.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        method : str, default="spearman"
            Linkage method to use for calculating clusters.
            See ``scipy.cluster.hierarchy.linkage()`` documentation for more information.

        clustering : bool, default=True
            Whether to apply a hierarchical clustering on the correlation matrix.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if phenotypeColumn is None:
            colName = [
                col for col in self.designMatrix.columns if "Intercept" not in col]
            if len(colName) == 1:
                phenotypeColumn = colName[0]
            else:
                raise RuntimeError("Provide phenotypeColumn")

        return plotSamplesCorrelation(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                                      pValueThreshold, lfcThreshold, method, clustering, path)

    def plotSamplesGenesCorrelation(self, transformedCounts=None, testResults=None, phenotypeData=None,
                                    phenotypeColumn=None, pValueThreshold=None, lfcThreshold=None,
                                    metric="correlation", path=None):
        """
        Plot a matrix of the correlations between samples and genes.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        phenotypeColumn : str
            Name of the column of the `phenotypeData` DataFrame
            which contains the data about the phenotype of interest.
            If not provided `DEGA` tries to retrive it from the design matrix,
            if it fails it raises an error.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        metric : str, default="correlation"
            Distance metric to use for the data.
            See ``scipy.spatial.distance.pdist()`` documentation for more options.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if phenotypeColumn is None:
            colName = [
                col for col in self.designMatrix.columns if "Intercept" not in col]
            if len(colName) == 1:
                phenotypeColumn = colName[0]
            else:
                raise RuntimeError("Provide phenotypeColumn")

        return plotSamplesGenesCorrelation(transformedCounts, testResults, phenotypeData,
                                           phenotypeColumn, pValueThreshold, lfcThreshold,
                                           metric, path)

    def plotPCAExplainedVariance(self, transformedCounts=None, testResults=None, phenotypeData=None,
                                 phenotypeColumn=None, pValueThreshold=None, lfcThreshold=None, path=None):
        """
        Plot the cumulative explained variance by the principal components.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        phenotypeColumn : str
            Name of the column of the `phenotypeData` DataFrame
            which contains the data about the phenotype of interest.
            If not provided `DEGA` tries to retrive it from the design matrix,
            if it fails it raises an error.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if phenotypeColumn is None:
            colName = [
                col for col in self.designMatrix.columns if "Intercept" not in col]
            if len(colName) == 1:
                phenotypeColumn = colName[0]
            else:
                raise RuntimeError("Provide phenotypeColumn")

        return plotPCAExplainedVariance(transformedCounts, testResults, phenotypeData,
                                        phenotypeColumn, pValueThreshold, lfcThreshold, path)

    def plotPCAComponentsDensity(self, transformedCounts=None, testResults=None,
                                 phenotypeData=None, phenotypeColumn=None, n_components=None,
                                 pValueThreshold=None, lfcThreshold=None, path=None):
        """
        Plot the density of the `n_components` first principal components.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        phenotypeColumn : str
            Name of the column of the `phenotypeData` DataFrame
            which contains the data about the phenotype of interest.
            If not provided `DEGA` tries to retrive it from the design matrix,
            if it fails it raises an error.

        n_components : int, optional
            Number of principal components to take into consideration.
            If not provided, it defaults to the number of samples.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        metric : str, default="correlation"
            Distance metric to use for the data.
            See ``scipy.spatial.distance.pdist()`` documentation for more options.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if phenotypeColumn is None:
            colName = [
                col for col in self.designMatrix.columns if "Intercept" not in col]
            if len(colName) == 1:
                phenotypeColumn = colName[0]
            else:
                raise RuntimeError("Provide phenotypeColumn")

        return plotPCAComponentsDensity(transformedCounts, testResults, phenotypeData,
                                        phenotypeColumn, n_components, pValueThreshold,
                                        lfcThreshold, path)

    def plotPCA(self, transformedCounts=None, testResults=None, phenotypeData=None,
                phenotypeColumn=None, components=("PC1", "PC2"), pValueThreshold=None,
                lfcThreshold=None, path=None):
        """
        Plot the results of a PCA for the `components`.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        phenotypeData : pandas.DataFrame, optional
            Matrix with at least a column describing the tested phenotypes.
            By default, the one provided when loading the dataset is used.

        phenotypeColumn : str
            Name of the column of the `phenotypeData` DataFrame
            which contains the data about the phenotype of interest.
            If not provided `DEGA` tries to retrive it from the design matrix,
            if it fails it raises an error.

        components : tuple, default=("PC1", "PC2")
            Principal components to take into consideration.
            It should be provided as a tuple of strings
            with the two names of the components.
            The name should be "PC" followed by the number of the component.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        metric : str, default="correlation"
            Distance metric to use for the data.
            See ``scipy.spatial.distance.pdist()`` documentation for more options.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold
        if phenotypeData is None:
            phenotypeData = self.phenotypeData
        if phenotypeColumn is None:
            colName = [
                col for col in self.designMatrix.columns if "Intercept" not in col]
            if len(colName) == 1:
                phenotypeColumn = colName[0]
            else:
                raise RuntimeError("Provide phenotypeColumn")

        return plotPCA(transformedCounts, testResults, phenotypeData, phenotypeColumn,
                       components, pValueThreshold, lfcThreshold, path)

    def plotCoexpressionNetwork(self, transformedCounts=None, testResults=None, lfcThreshold=None,
                                pValueThreshold=None, correlationMethod="spearman",
                                correlationThreshold=None, onlyPositiveCorrelation=False, seed=12345,
                                labels=True, fontSize=1, nodeAlpha=0.9, edgeAlpha=0.4, path=None):
        """
        Plot a coexpression network of significant genes.
        Node size represents the statistical validity (-log10 p-value).
        Nodes of up-regulated genes are in red and those of down-regulated ones are in blue.
        Edge color illustrates the positive (red) or negative (blue) correlation between genes.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        correlationMethod : str, default="spearman"
            Linkage method to use for calculating clusters.
            See ``scipy.cluster.hierarchy.linkage()`` documentation for more information.

        correlationThreshold : float, optional
            A value which specifies a threshold for the correlation.
            If not specified it defaults to the 0.975 quantile of the correlations.

        onlyPositiveCorrelation : bool, default=Fale
            Whether to keep only positive correlations.

        seed : int, default=12345
            Set the random state for deterministic node layouts.
            It is the seed used by the random number generator.

        labels : bool, default=True
            Whether to plot the labels of the nodes.

        fontsize : int, default=1
            Font size for text labels.

        nodeAlpha : float, default=0.9
            The node transparency. This can be a single alpha value,
            in which case it will be applied to all the nodes of color.
            Otherwise, if it is an array,
            the elements of alpha will be applied to the colors in order
            (cycling through alpha multiple times if necessary).

        edgeAlpha : float, default=0.4
            The edge transparency

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        if lfcThreshold is None:
            lfcThreshold = self.lfcThreshold

        return plotCoexpressionNetwork(transformedCounts, testResults, lfcThreshold, pValueThreshold,
                                       correlationMethod, correlationThreshold, onlyPositiveCorrelation,
                                       seed, labels, fontSize, nodeAlpha, edgeAlpha, path)

    def plotIndependentFiltering(self, testResults=None, baseMean=None,
                                 pValueThreshold=None, method="fdr_bh", path=None):
        """
        Technical plot about the independent filtering.
        Displays the number of rejections over the quantiles of the base mean.
        See https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#indfilttheory

        Parameters
        ----------
        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        baseMean : numpy.ndarray(dtype=float, ndim=1), optional
            Row means of normalized counts.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        method : str, default="fdr_bh"
            The method to use for adjusting p-values for multiple tests.
            Check `method` parameter of the function `multipletests`
            from from `statsmodels.stats.multitest`.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if testResults is None:
            testResults = self._testResults()
        if baseMean is None:
            baseMean = self.baseMean
        if pValueThreshold is None:
            pValueThreshold = self.alpha
        return plotIndependentFiltering(
            testResults, baseMean, pValueThreshold, method, path)

    def plotCooks(self, testResults=None, baseMean=None, path=None):
        """
        Technical plot about the independent filtering.
        Displays the number of rejections over the quantiles of the base mean.
        See https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#indfilttheory

        Parameters
        ----------
        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        baseMean : numpy.ndarray(dtype=float, ndim=1), optional
            Row means of normalized counts.

        path : str, optional
            Path with filename and extension where to save the plot.
            If `None` (default), the plot is not saved.
        """
        if testResults is None:
            testResults = self._testResults()
        if baseMean is None:
            baseMean = self.baseMean
        return plotCooks(testResults, baseMean, path)

    def transformCounts(self, method="vst", counts=None, normalizedCounts=None, sizeFactors=None,
                        baseMean=None, dispGeneEsts=None, dispFit=None, fitType=None, fitParams=None, fitFunc=None,
                        minDisp=1e-8, intercept=None, betaPriorVar=None, shift=1):
        """
        Differential expression analysis is carried out on raw counts.
        However, for other downstream analyses
        (e.g., for visualizations or machine learning applications)
        it might be useful to work with transformed versions
        (variance stabilizing transformation (default),
        regularized logarithm and shifted logarithm)
        of the count data.

        Parameters
        ----------
        method : {"vst", "rlog", "shiftedLog"}
            Transformation method.
            "vsr" for variance stabilizing,
            "rlog" for regularized logarithm,
            "shiftedLog" for shifted logarithm.

        counts : pandas.DataFrame, optional
            Matrix of the read counts (non-negative integers).
            By default, the one provided as `geneCounts`
            when loading the dataset is used.

        normalizedCounts : pandas.DataFrame, optional
            Matrix of normalized counts.
            By default, the ones normalized
            dividing the `counts` by the `sizeFactors`
            when loading the dataset is used.

        sizeFactors : numpy.ndarray(dtype=float, ndim=1), optional
            Multiplicative normalizing factors for each sample.

        baseMean : numpy.ndarray(dtype=float, ndim=1), optional
            Row means of normalized counts.

        dispGeneEsts : numpy.ndarray(dtype=float, ndim=1), optional
            Gene-wise dispersion estimates.
            If not provided, defaults to the estimates of the analysis.

        dispFit : numpy.ndarray(dtype=float, ndim=1), optional
            Fitted dispersions. If not provided,
            defaults to the estimates of the analysis.

        fitType : {"parametric", "local", "mean", "kernel"}, optional
            Either "parametric", "local", "mean", or "kernel"
            for the type of fitting of dispersions to the mean intensity.
            If not provided, it defaults to the `fitType` used in the analysis.

        fitParams : dict, optional
            Required if `method` is `vst`. The parameters of the fitting
            of dispersions to the mean intensity.
            If not provided, it defaults to the fitting parameters of the analysis.
        
        fitParams : func, optional
            Required if `method` is `vst` and `fitType` is `local`. The function uset to fit
            the dispersions to the mean intensity.
            If not provided, it defaults to the fitting function of the analysis.

        minDisp : float, default=1e-8
            Small value for the minimum dispersion, to allow for calculations in log scale,
            one order of magnitude above this value is used
            as a test for inclusion in mean-dispersion fitting.

        intercept: np.ndarray(dtype=int, ndim=1), optional
            Required if `method` is `rlog`.
            If not provided, it defaults to the intercept column of the design matrix.

        betaPriorVar : numpy.ndarray(dtype=float, ndim=1), optional
            A vector with length equal to the number of design parameters
            including the intercept.
            `betaPriorVar` gives the variance of the prior on the sample betas
            on the log2 scale.
            By default, it is estimated from the data.

        shift : float, default=1
            A positive constant to be added to the counts in order to avoid
            an error raised by trying to compute the logarithm of zero.
        """
        if counts is None:
            counts = self.counts
        if normalizedCounts is None:
            normalizedCounts = self.normalizedCounts
        if sizeFactors is None:
            sizeFactors = self.sizeFactors
        if baseMean is None:
            baseMean = self.baseMean
        if dispGeneEsts is None:
            dispGeneEsts = self._dispersionsResults()["dispGeneEst"].values
        if dispFit is None:
            dispFit = self._dispersionsResults()["dispFit"].values
        if fitType is None:
            fitType = self._fitType
        if fitParams is None:
            fitParams = self._fitParams
        if fitFunc is None:
            fitFunc = self.__dispFitFunc
        if method == "vst":
            self.__transformedCounts = vst(
                normalizedCounts, sizeFactors, baseMean, dispGeneEsts, fitType, fitParams, minDisp=1e-8, fitFunc=fitFunc)
        elif method == "rlog":
            self.__transformedCounts = rlog(
                counts, normalizedCounts, sizeFactors, baseMean, dispFit, intercept=None, betaPriorVar=None)
        elif method == "shiftedLog":
            self.__transformedCounts = shiftedLog(
                normalizedCounts, shift=shift)
        else:
            raise RuntimeError(
                f"{method} method for transforming/scaling counts not implemented, choose one of 'vst', 'rlog' and 'shiftedLog'")
        return self._transformedCounts

    def coexpressionNetwork(self, transformedCounts=None, testResults=None, lfcThreshold=1,
                            pValueThreshold=0.01, correlationMethod="spearman"):
        """
        Adjacency matrix of a coexpression network of significant genes.

        Parameters
        ----------
        transformedCounts : pandas.DataFrame, optional
            Differential expression analysis is carried out on raw counts.
            However, for other downstream analyses
            (e.g., for visualizations or machine learning applications)
            it might be useful to work with transformed versions
            (variance stabilizing transformation (default),
            regularized logarithm and shifted logarithm)
            of the count data.
            If `transformedCounts` is not provided, `DEGA`
            will apply a variance stabilizing transformation automatically
            in background.

        testResults : pandas.DataFrame, optional
            Results of the statistical test.

        pValueThreshold : float, optional
            A non-negative value which specifies a threshold for the p-value.
            If not specified defaults to `alpha`.

        lfcThreshold : float, optional
            A non-negative value which specifies a threshold for the log2 fold change.
            If not specified defaults to `lfcThreshold`.

        correlationMethod : str, default="spearman"
            Linkage method to use for calculating clusters.
            See ``scipy.cluster.hierarchy.linkage()`` documentation for more information.
        """
        if transformedCounts is None:
            try:
                transformedCounts = self._transformedCounts
            except:
                transformedCounts = self.transformCounts()
        if testResults is None:
            testResults = self._testResults()
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
        return adjacencyMatrix

    @property
    def results(self):
        """
        Results of the differential expression analysis.
        """
        return self._results(showIntercept=False, showAllZero=False)

    def _results(self, showIntercept=False, showAllZero=False):
        try:
            if showAllZero:
                results = pd.concat([self.meanVarZero, self._dispersionsResults(
                ), self._testResults(showIntercept)], axis=1)
                allResults = pd.DataFrame(
                                np.tile(np.concatenate(
                                    [np.array([0, 0, True]), np.full(
                                        len(results.columns)-3, np.NaN)]
                                ), (self.geneCounts.shape[0], 1)).astype("object"),
                            index=self.geneCounts.index, columns=results.columns)
                allResults.loc[results.index] = results
                return allResults
            else:
                return pd.concat([self.meanVarZero, self._dispersionsResults(), self._testResults(showIntercept)], axis=1)
        except:
            raise RuntimeError("Run analsis first")

    # not a property for coherence with _testResults
    def _dispersionsResults(self):
        try:
            return self.__dispersionsResults.copy()
        except:
            raise RuntimeError("Run dispersions analsis first")

    # not a property because otherwise the showIntercept parameter could not be passed
    def _testResults(self, showIntercept=False):
        try:
            if showIntercept:
                return self.__testResults.copy()
            else:
                return self.__testResults[[
                    col for col in self.__testResults.columns if "Intercept" not in col]].copy()
        except:
            raise RuntimeError("Test statistical significance first")

    @property
    def summary(self):
        """
        A brief summary of info about the dataset.
        """
        log.info(self)

    @property
    def significantGenes(self):
        """
        Genes having an adjusted p-value below `pValueThreshold` and
        a absolute log2 fold change higher than `lfcThreshold`.
        """
        LFCColumnName = [col for col in self.results.columns if (
            col.startswith("log2 fold change") and "Intercept" not in col)][0]
        PvalueColumnName = [col for col in self.results.columns if (
            col.startswith("adjusted p-value") and "Intercept" not in col)][0]
        significantGenes = self.results[(np.abs(self.results[LFCColumnName]) > self.lfcThreshold) & (
            self.results[PvalueColumnName] < self.pValueThreshold)]
        return significantGenes

    @property
    def geneCounts(self):
        """
        Matrix of the read counts (non-negative integers).
        Columns of geneCounts correspond to rows of phenotypeData.
        """
        return self.__geneCounts

    @property
    def phenotypeData(self):
        """
        Matrix with at least a column describing the tested phenotypes.
        Rows of phenotypeData correspond to columns of geneCounts.
        """
        return self.__phenotypeData

    @property
    def counts(self):
        """
        Matrix of the read counts (non-negative integers).
        By default, the one provided as `geneCounts`
        when loading the dataset is used.
        """
        return self.__counts

    @property
    def normalizedCounts(self):
        """
        Matrix of normalized counts.
        By default, the ones normalized
        dividing the `counts` by the `sizeFactors`
        when loading the dataset is used.
        """
        return self.__normalizedCounts

    @property
    def genes(self):
        """
        numpy.ndarray of genes in `geneCounts`.
        """
        return self.__genes

    @property
    def upregulatedGenes(self):
        """
        Genes having an adjusted p-value below `pValueThreshold` and
        a log2 fold change higher than `lfcThreshold`.
        """
        try:
            return self.__upregulatedGenes
        except:
            raise RuntimeError("Run analsis first")

    @property
    def downregulatedGenes(self):
        """
        Genes having an adjusted p-value below `pValueThreshold` and
        a log2 fold change lower than `-lfcThreshold`
        """
        try:
            return self.__downregulatedGenes
        except:
            raise RuntimeError("Run analsis first")

    @property
    def designFormula(self):
        """
        Design Formula
        """
        return self.__designFormula

    @property
    def designMatrix(self):
        """
        Design Matrix.
        A matrix of values of explanatory variables of the samples.
        """
        return self.__designMatrix

    @property
    def sizeFactors(self):
        """
        Multiplicative normalizing factors for each sample.
        """
        return self.__sizeFactors

    @property
    def weights(self):
        """
        2D array of weights.
        """
        return self.__weights

    @property
    def useWeights(self):
        """
        Whether to use provided weights.
        """
        return self.__useWeights

    @property
    def test(self):
        """
        The used type of significance test.
        Either "Wald" or "LRT":
        Wald significance tests or the likelihood ratio test
        on the difference in deviance between a full and reduced design formula.
        """
        return self.__test

    @property
    def alpha(self):
        """
        A non-negative value which specifies a threshold for the p-value.
        """
        return self.__alpha

    @property
    def lfcThreshold(self):
        """
        A non-negative value which specifies a threshold for the log2 fold change.
        """
        return self.__lfcThreshold

    @property
    def meanVarZero(self):
        """
        DataFrame with row means and variances of normalized counts.
        Has a boolean column specifying if the row sums are zero.
        """
        return self.__meanVarZero

    @property
    def baseMean(self):
        """
        Row means of normalized counts.
        """
        return self.__baseMean

    @property
    def baseVar(self):
        """
        Row variance of normalized counts.
        """
        return self.__baseVar

    @property
    def withShrinkage(self):
        """
        Whether the log2 fold changes and standard error were shrunken.
        """
        return self.__withShrinkage

    @property
    def _allZero(self):
        """
        Boolean array specifying if the row sums are zero.
        """
        return self.__allZero

    @property
    def _dispGeneEstResults(self):
        return self.__dispGeneEstResults

    @property
    def _dispFitResults(self):
        return self.__dispFitResults

    @property
    def _dispMAPResults(self):
        return self.__dispMAPResults

    @property
    def _fitType(self):
        return self.__fitType

    @property
    def _dispFitFunc(self):
        return self.__dispFitFunc

    @property
    def _fitParams(self):
        return self.__fitParams

    @property
    def _cooks(self):
        return pd.DataFrame(self.__cooks, index=self.counts.index, columns=self.counts.columns)

    @property
    def _mu(self):
        return self.__mu

    @property
    def _betaPriorVar(self):
        return self.__betaPriorVar

    @property
    def _replacedOutliersCounts(self):
        return self.__replacedOutliersCounts

    @property
    def _transformedCounts(self):
        return self.__transformedCounts

    def __name__(self):
        return "DEGA Dataset"

    def __repr__(self):
        s = "DEGA Dataset\n"
        s += f"Genes with non-zero read counts: {len(self.genes)}\n"
        s += f"Number of samples: {self.designMatrix.shape[0]}\n"
        try:
            LFCColumnName = [col for col in self.results.columns if (
                col.startswith("log2 fold change") and "Intercept" not in col)][0]
            PvalueColumnName = [col for col in self.results.columns if (
                col.startswith("p-value") and "Intercept" not in col)][0]
            AdjustedPvalueColumnName = [col for col in self.results.columns if (
                col.startswith("adjusted p-value") and "Intercept" not in col)][0]
            significant = self.results[(np.abs(self.results[LFCColumnName]) > self.lfcThreshold) & (
                self.results[AdjustedPvalueColumnName] < self.alpha)][[LFCColumnName, AdjustedPvalueColumnName]]
            significant = significant[~np.isnan(significant)]
            up = (significant[LFCColumnName] > self.lfcThreshold).sum()
            down = (significant[LFCColumnName] < -self.lfcThreshold).sum()
            outliers = ((self.baseMean > 0) & np.isnan(
                self.results[PvalueColumnName].values)).sum()
            low_counts = ((~np.isnan(self.results[PvalueColumnName].values)) & np.isnan(
                self.results[AdjustedPvalueColumnName].values)).sum()
            notAllZero = (self.baseMean > 0).sum()
            s += f"Adjusted P-Value<{self.alpha}, LFC<{self.lfcThreshold}\n"
            s += f"Upregulated: {up}, {round((up/notAllZero)*100,2)}%\n"
            s += f"Downregulated: {down}, {round((down/notAllZero)*100,2)}%\n"
            s += f"Outliers: {outliers}, {round((outliers/notAllZero)*100,2)}%\n"
            s += f"Low counts: {low_counts}, {round((low_counts/notAllZero)*100,2)}%\n"
            return s
        except:
            return s
