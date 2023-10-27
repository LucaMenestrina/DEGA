# Switch genes IDEntifier
# For more info, check:
# https://academic.oup.com/plcell/article/26/12/4617/6102410
# https://www.nature.com/articles/srep44797
# https://academic.oup.com/bioinformatics/article/38/2/586/6370739


import numpy as np
import pandas as pd

import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats
from statsmodels.stats.multitest import multipletests


class SIDE:
    """
    SIDE (Switch genes IDEntifier) module of DEGA

    Parameters
    ----------
    expressionData : pandas.DataFrame
        Matrix of the counts
    
    significantGenes : numpy.ndarray
        Array of significant genes determined by differential expression analysis

    """
    def __init__(self, expressionData, significantGenes):
        """
        Initialize self

        Loads expressionData and significantGenes
        """
        self.__expressionData = expressionData.loc[significantGenes]
        self.__significantGenes = significantGenes
    
    def identifySwitchGenes(self, expressionData=None, significantGenes=None, correlationMethod="pearson",
                        thresholdType="correlation", completeScan=False, multipletestsCorrectionMethod="fdr_bh", significanceThreshold=0.05,
                        correlationThreshold=None, ratioLargestComponent=0.99, clusteringMethod="greedy_modularity", numClusters=None, randomSeed=12345):
        """
        Main function for identifing the switch genes.

        Parameters
        ----------
        expressionData : pandas.DataFrame, optional
            Matrix of the counts
        
        significantGenes : numpy.ndarray, optional
            Array of significant genes determined by differential expression analysis
        
        correlationMethod : string, default="pearson"
            Correlation method for building the correlation matrix.
            See ``pandas.DataFrame.corr`` documentation for more information.

        thresholdType : {"correlation", "significance"}, default="correlation"
            Either "correlation" or "significance", for the type of threshold to be used in filtering the correlation matrix:
            Correlation performs an integrity test and keeps the highest correlation threshold leading to a fully connected network.
            Significance performs a double-sided t-test and keeps all values significant considering ```significanceThreshold``` as reference value

        completeScan : bool, default=False
            Whether to perform a full correlation scan in the integrity test
            or limit it between the 0.5 and 0.99 quantiles of the correlations distribution
            Only used if ```thresholdType == "correlation"```

        multipletestsCorrectionMethod : string, default="fdr_bh"
            Method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Available methods are:
                bonferroni : one-step correction
                sidak : one-step correction
                holm-sidak : step down method using Sidak adjustments
                holm : step-down method using Bonferroni adjustments
                simes-hochberg : step-up method (independent)
                hommel : closed method based on Simes tests (non-negative)
                fdr_bh : Benjamini/Hochberg (non-negative)
                fdr_by : Benjamini/Yekutieli (negative)
                fdr_tsbh : two stage fdr correction (non-negative)
                fdr_tsbky : two stage fdr correction (non-negative)
            See statsmodels.stats.multitest.multipletest for more
        
        significanceThreshold : float, default=0.05
            Value for determining the significance of the correlations
            Only used if ```thresholdType == "significance"```
        
        correlationThreshold : float, optional
            Value for filtering the correlation matrix.
            If set, the integrity test will not be performed and this value will be used instead
        
        ratioLargestComponent : float, default=0.99
            Minimum ratio of nodes in the largest component.
            The integrity test is stopped if the largest component
            contains fewer than ```ratioLargestComponent``` of the total nodes.

        clusteringMethod : {"greedy_modularity", "kmeans"}, default="greedy_modularity"
            Clustering method for identifing network communities

        numClusters : int, optional
            Number of clusters
            Only used if ```clusteringMethod == "kmeans"```

        randomSeed : int, default=12345
            Set the random state for kmeans clustering.
            It is the seed used by the random number generator.
            Only used if ```clusteringMethod == "kmeans"```

        """
        if significantGenes is None:
            significantGenes = self.__significantGenes
        else:
            self.__significantGenes = significantGenes
        if expressionData is None:
            expressionData = self.__expressionData
        else:
            expressionData = expressionData.loc[significantGenes]
            self.__expressionData = expressionData
        self.__correlationMethod = correlationMethod
        self.__correlationMatrix = self._compute_correlation(expressionData, correlationMethod=correlationMethod)
        # if thresholdType == "correlation":
        #     self.__correlationThreshold = self._get_correlationThreshold(self.__correlationMatrix, completeScan=completeScan)
        #     self.__correlationMatrix[abs(self.__correlationMatrix) <= self.__correlationThreshold] = 0
        # elif thresholdType == "significance":
        #     self.__correlationMatrix = self._get_significant_correlations(self.__correlationMatrix, expressionData.shape[1], significanceThreshold=significanceThreshold)
        # self.__correlationNetwork = nx.from_pandas_adjacency(self.__correlationMatrix)
        self.__correlationMatrix, self.__correlationThreshold = self._filter_correlations(self.__correlationMatrix, expressionData.shape[1], multipletestsCorrectionMethod=multipletestsCorrectionMethod, significanceThreshold=significanceThreshold, correlationThreshold=correlationThreshold, ratioLargestComponent=ratioLargestComponent, completeScan=completeScan)
        self.__clusters = self._get_clusters(self.__correlationMatrix, method=clusteringMethod, numClusters=numClusters, randomSeed=randomSeed)
        self.__apcc = self._get_apcc(self.__correlationMatrix)
        self.__k, self.__ki = self._get_degrees(self.__correlationMatrix, self.__clusters)
        self.__cc = self._get_clusterphobic_coefficient(self.__k, self.__ki)
        self.__km, self.__ks, self.__z = self._get_normalized_within_module_degree(self.__correlationMatrix, self.__clusters, self.__ki, return_module_data=True)

        self.__switchGenes = np.array(significantGenes[(self.__cc > 0.8) & (self.__z < 2.5) & (self.__apcc < 0)])

        return self.switchGenes

    def _compute_correlation(self, expressionData, correlationMethod="pearson"):
        """
        Compute correlation matrix

        Parameters
        ----------
        expressionData : pandas.DataFrame
            Matrix of the counts
        
        correlationMethod : string, default="pearson"
            Correlation method for building the correlation matrix.
            See ``pandas.DataFrame.corr`` documentation for more information.
        
        """
        if correlationMethod == "pearson":
            correlationMatrix = pd.DataFrame(np.corrcoef(expressionData.values), columns=expressionData.index, index=expressionData.index)
        else:
            correlationMatrix = expressionData.T.corr(method=correlationMethod)
        np.fill_diagonal(correlationMatrix.values, 0)
        correlationMatrix[np.isnan(correlationMatrix)] = 0

        return correlationMatrix

    def _filter_correlations(self, correlationMatrix, n_samples, multipletestsCorrectionMethod="fdr_bh", significanceThreshold=0.05, correlationThreshold=None, ratioLargestComponent=0.99, completeScan=False):
        """
        Filter the correlationMatrix
        significant correlations determined by a t-test
        performs an integrity test in order to determine a correlation threshold

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations
        
        n_samples : int
            Number of samples in expressionData
        
        multipletestsCorrectionMethod : string, default="fdr_bh"
            Method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Available methods are:
                bonferroni : one-step correction
                sidak : one-step correction
                holm-sidak : step down method using Sidak adjustments
                holm : step-down method using Bonferroni adjustments
                simes-hochberg : step-up method (independent)
                hommel : closed method based on Simes tests (non-negative)
                fdr_bh : Benjamini/Hochberg (non-negative)
                fdr_by : Benjamini/Yekutieli (negative)
                fdr_tsbh : two stage fdr correction (non-negative)
                fdr_tsbky : two stage fdr correction (non-negative)
            See statsmodels.stats.multitest.multipletest for more
        
        significanceThreshold : float, default=0.05
            Value for determining the significance of the correlations
        
        correlationThreshold : float, optional
            Value for filtering the correlation matrix.
            If set, the integrity test will not be performed and this value will be used instead
        
        ratioLargestComponent : float, default=0.99
            Minimum ratio of nodes in the largest component.
            The integrity test is stopped if the largest component
            contains fewer than ```ratioLargestComponent``` of the total nodes.
        
        completeScan : bool, default=False
            Whether to perform a full correlation scan in the integrity test
            or limit it between the 0.5 and 0.99 quantiles of the correlations distribution
            Only used if ```thresholdType == "correlation"```
        
        """
        # statistically significant correlations (t-test)
        # n_samples = expressionData.shape[1]
        flat_indices = np.triu_indices_from(correlationMatrix.values, k=1)
        t_values = (correlationMatrix.values[flat_indices] * np.sqrt(n_samples-2))/(np.sqrt(1-correlationMatrix.values[flat_indices]**2))
        p_values = stats.t.sf(abs(t_values), df=n_samples-2)*2 # p-value two-tailed test
        _, adj_pvalues_flat, _, _ = multipletests(p_values, method=multipletestsCorrectionMethod, alpha=significanceThreshold)  # p_values correction for multiple tests
        adj_pvalues = np.zeros_like(correlationMatrix.values)
        adj_pvalues[flat_indices] = adj_pvalues_flat
        adj_pvalues = adj_pvalues + adj_pvalues.T
        np.fill_diagonal(adj_pvalues, np.inf)
        # adj_pvalues = pd.DataFrame(adj_pvalues, columns=expressionData.index, index=expressionData.index)
        correlationMatrix[np.isnan(correlationMatrix)] = 0
        correlationMatrix[adj_pvalues >= significanceThreshold] = 0  # test
        
        # def get_number_of_components(correlationMatrix, threshold):
        #     tmp_adj = correlationMatrix.copy()
        #     tmp_adj[abs(tmp_adj) <= threshold] = 0
        #     G = nx.from_pandas_adjacency(tmp_adj)
        #     ncc = nx.number_connected_components(G)
        #     return ncc
        def get_ratio_in_largest_component(correlationMatrix, threshold):
            tmp_adj = correlationMatrix.copy()
            tmp_adj[abs(tmp_adj) <= threshold] = 0
            G = nx.from_pandas_adjacency(tmp_adj)
            ratio = max([len(c) for c in nx.connected_components(G)])/nx.number_of_nodes(G)
            return ratio

        if correlationThreshold is None:
            if not completeScan:
                lb = np.quantile(abs(correlationMatrix.values.flatten()), 0.5).round(3)  # lower bound
                ub = np.quantile(abs(correlationMatrix.values.flatten()), 0.99).round(3)  # upper bound
                # check it's still single component, otherwise perfom a complete scan
                if get_ratio_in_largest_component(correlationMatrix, np.min(correlationMatrix.values[(correlationMatrix.values > lb)])) < ratioLargestComponent:
                    lb = correlationMatrix.values.flatten().min().round(3)  # lower bound
                    ub = correlationMatrix.values.flatten().max().round(3)  # upper bound
            else:
                lb = correlationMatrix.values.flatten().min().round(3)  # lower bound
                ub = correlationMatrix.values.flatten().max().round(3)  # upper bound
            thresholds = np.unique(correlationMatrix.values[(correlationMatrix.values > lb) & (correlationMatrix.values < ub)].round(3))
            correlationThreshold = thresholds[0]
            # ncc = 1  # number of connected components
            # for threshold in tqdm(thresholds, desc="determine correlation threshold"):
            #     ncc = get_number_of_components(correlationMatrix, threshold)
            #     if ncc == 1:
            #         correlationThreshold = threshold
            #     else:
            #         break
            ratio = 1  # number of nodes in largest connected component
            for threshold in tqdm(thresholds, desc="determine correlation threshold"):
                ratio = get_ratio_in_largest_component(correlationMatrix, threshold)
                if ratio >= ratioLargestComponent:
                    correlationThreshold = threshold
                else:
                    break
            correlationMatrix[abs(correlationMatrix) <= correlationThreshold] = 0
        else:
            correlationMatrix[abs(correlationMatrix) <= correlationThreshold] = 0

        return correlationMatrix, correlationThreshold

    def _get_clusters(self, correlationMatrix, method="greedy_modularity", numClusters=None, randomSeed=12345):
        """
        Identify network communities
        If ```method="kmeans"``` ```numClusters``` must be defined

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations

        clusteringMethod : {"greedy_modularity", "kmeans"}, default="greedy_modularity"
            Clustering method for identifing network communities

        numClusters : int, optional
            Number of clusters
            Only used if ```clusteringMethod == "kmeans"```

        randomSeed : int, default=12345
            Set the random state for kmeans clustering.
            It is the seed used by the random number generator.
            Only used if ```clusteringMethod == "kmeans"```

        """
        if method == "greedy_modularity":
            correlationNetwork = nx.from_pandas_adjacency(correlationMatrix)
            node2community = {node:n for n, comm in enumerate(nx.community.greedy_modularity_communities(correlationNetwork)) for node in comm}
            clusters = np.array([node2community[node] for node in correlationNetwork.nodes()])
        elif method == 'kmeans':
            if numClusters is None:
                raise Exception("Number fo clusters not defined")
            else:
                correlationMatrix = correlationNetwork.to_pandas_adjacency()
                clusters = KMeans(n_clusters=numClusters, random_state=randomSeed, n_init=correlationMatrix.shape[0]).fit(correlationMatrix).labels_

        return clusters

    def _get_apcc(self, correlationMatrix):
        """
        Compute Average Pearson Correlation Coefficient

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations

        """
        apcc = correlationMatrix[correlationMatrix != 0].mean(axis=1)

        return apcc

    def _get_degrees(self, correlationMatrix, clusters):
        """
        Compute the degree and the within-cluster degree

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations

        clusters : numpy.array
            Ordered array with the number identifing the cluster for every node

        """
        k = (correlationMatrix != 0).sum()
        ki = ((clusters[:, np.newaxis] == clusters[np.newaxis, :]) & (correlationMatrix != 0))  # number of edges a node has with nodes in its module
        np.fill_diagonal(ki.values, 0)  # not consider self-edges
        ki = ki.sum(axis=1)

        return k, ki

    def _get_clusterphobic_coefficient(self, k, ki):
        """
        Compute the clusterphobic coefficient

        Parameters
        ----------
        k : numpy.array
            Nodes degree
        ki : numpy.array
            Nodes within-cluster degree
            (number of edges a node has with nodes in its module)

        """
        cc = 1 - (ki/k)**2

        return cc

    def _get_normalized_within_module_degree(self, correlationMatrix, clusters, ki, return_module_data=False):
        """
        Normalize the within-module degree

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations

        clusters : numpy.array
            Ordered array with the number identifing the cluster for every node

        ki : numpy.array
            Nodes within-cluster degree
            (number of edges a node has with nodes in its module)
        
        return_module_data : bool, default=False
            Whether to return the mean of degrees in each module and the standard deviation of degrees in each module

        """
        km = np.zeros(correlationMatrix.shape[0])  # mean of degrees in each module
        ks = np.zeros(correlationMatrix.shape[0])  # standard deviation of degrees in each module
        for n in set(clusters):
            where = (clusters == n)
            km[where] = np.mean((correlationMatrix[where] != 0).sum(axis=1))
            ks[where] = np.std((correlationMatrix[where] != 0).sum(axis=1))
        z = (ki - km)/ks  # within-module degree

        if return_module_data:
            return km, ks, z
        else:
            return z

    def plot_correlation_distribution(self, correlationMatrix=None, kde=True, bins=100):
        """
        Plot the distribution of correlations (histogram)

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame, optional
            Matrix of the gene expression correlations
        
        kde : bool, default=True
            Whether to plot also the kernel density estimation
        
        bins : str, number, vector, or a pair of such values, default=100
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to ```numpy.histogram_bin_edges().```
            Check ```seaborn.histplot``` for more.

        """
        if correlationMatrix is None and hasattr(self, "correlationMatrix"):
            correlationMatrix = self.correlationMatrix
        else:
            raise Exception("define 'correlationMatrix'")
        return sns.displot(correlationMatrix.values.flatten(), kde=kde, bins=bins)


    def plot_kmeans_sse(self, correlationMatrix=None, randomSeed=12345):
        """
        Scree plot of the sum of square error for kmeans clustering
        A good choice for the optimal number of clusters is in correspondence of the elbow

        Parameters
        ----------
        correlationMatrix : pandas.DataFrame
            Matrix of the gene expression correlations

        randomSeed : int, default=12345
            Set the random state for kmeans clustering.
            It is the seed used by the random number generator.
            Only used if ```clusteringMethod == "kmeans"```

        """
        if correlationMatrix is None and hasattr(self, "correlationMatrix"):
            correlationMatrix = self.correlationMatrix
        else:
            raise Exception("define 'correlationMatrix'")
        d = {n:KMeans(n_clusters=n, random_state=randomSeed, n_init="auto").fit(correlationMatrix).inertia_ for n in range(1,correlationMatrix.shape[0]+1)}
        sns.scatterplot(x=d.keys(), y=d.values())

    def plot_apcc(self, apcc=None, hubs=True, k=None, kde=True, bins=100):
        """
        Plot the distribution of the Average Pearson Correlation Coefficients (histogram)

        Parameters
        ----------
        apcc : numpy.array, optional
            Array of Average Pearson Correlation Coefficients
        
        hubs : bool, default=True
            Whether to plot the distribution only for hub nodes (degree > 5)

        k : np.array, optional
            Array of node degrees
            Only used if ```hubs=True```
        
        kde : bool, default=True
            Whether to plot also the kernel density estimation
        
        bins : str, number, vector, or a pair of such values, default=100
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to ```numpy.histogram_bin_edges().```
            Check ```seaborn.histplot``` for more.

        """
        if apcc is None and hasattr(self, "_apcc"):
            apcc = self._apcc
        else:
            raise Exception("define 'apcc'")
        if hubs:
            if k is None and hasattr(self, "_k"):
                k = self._k
            else:
                raise Exception("define 'k'")
            sns.displot(apcc[k>5], kde=kde, bins=bins)
        else:
            sns.displot(apcc, kde=kde, bins=bins)

    def plot_degree_distribution(self, k=None, kde=True, bins=100):
        """
        Plot the distribution of the node degrees (histogram)

        Parameters
        ----------
        k : np.array, optional
            Array of node degrees`
        
        kde : bool, default=True
            Whether to plot also the kernel density estimation
        
        bins : str, number, vector, or a pair of such values, default=100
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to ```numpy.histogram_bin_edges().```
            Check ```seaborn.histplot``` for more.

        """
        if k is None and hasattr(self, "_k"):
            k = self._k
        else:
            raise Exception("define 'k'")
        sns.displot(k, kde=kde, bins=bins)

    def plot_heatmap(self, cc=None, z=None, apcc=None):
        """
        Plot the Heat Cartography Map
        For more information check https://academic.oup.com/plcell/article/26/12/4617/6102410

        Parameters
        ----------
        cc : numpy.array, optional
            Array of Clusterphobic coefficient

        z : numpy.array, optional
            Array of Normalized within-cluster degree

        apcc : numpy.array, optional
            Array of Average Pearson Correlation Coefficients

        """
        if cc is None and hasattr(self, "_cc"):
            cc = self._cc
        else:
            raise Exception("define 'cc'")
        if z is None and hasattr(self, "_z"):
            z = self._z
        else:
            raise Exception("define 'z'")
        if apcc is None and hasattr(self, "_apcc"):
            apcc = self._apcc
        else:
            raise Exception("define 'apcc'")
        plot = plt.scatter(x=cc, y=z, c=apcc, cmap="seismic", s=25, vmin=-1, vmax=1)  # RdYlBu_r
        plt.axhline(y=2.5, c="darkgrey")
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.vlines(x=[0, 0.625, 0.8], ymin=ymin, ymax=2.5, color="darkgrey")
        plt.vlines(x=[0.3, 0.75], ymin=2.5, ymax=ymax, color="darkgrey")
        # limits could be changed after plotting lines
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.xlabel(r"Clusterphobic Coefficient ($K_{\pi}$)")
        plt.ylabel(r"Within-Module Degree ($z_g$)")
        plt.text(x=xmin/2, y=2.45, s="R1", horizontalalignment="center", verticalalignment="top")
        plt.text(x=0.625/2, y=2.45, s="R2", horizontalalignment="center", verticalalignment="top")
        plt.text(x=(0.625+0.8)/2, y=2.45, s="R3", horizontalalignment="center", verticalalignment="top")
        plt.text(x=(0.8+xmax)/2, y=2.45, s="R4", horizontalalignment="center", verticalalignment="top")
        plt.text(x=(xmin+0.3)/2, y=2.55, s="R5", horizontalalignment="center", verticalalignment="bottom")
        plt.text(x=(0.3+0.75)/2, y=2.55, s="R6", horizontalalignment="center", verticalalignment="bottom")
        plt.text(x=(0.75+xmax)/2, y=2.55, s="R7", horizontalalignment="center", verticalalignment="bottom")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("APCC")

    @property
    def expressionData(self):
        """
        Gene Expression Data
        Gene Counts
        """
        return self.__expressionData

    @property
    def switchGenes(self):
        """
        Switch Genes
        """
        return self.__switchGenes

    @property
    def _correlationMethod(self):
        """
        Correlation Method
        Used for building the correlation matrix
        """
        return self.__correlationMethod

    @property
    def correlationMatrix(self):
        """
        Correlation Matrix
        Correlations between genes in ```expressionData```
        """
        return self.__correlationMatrix

    # @property
    # def correlationNetwork(self):
    #     return self.__correlationNetwork

    @property
    def _correlationThreshold(self):
        """
        Correlation Threshold for filtering ```correlationMatrix```
        """
        return self.__correlationThreshold

    @property
    def _clusters(self):
        """"
        Ordered array with the number identifing the cluster for every node
        """
        return self.__clusters

    @property
    def _apcc(self):
        """
        Array of Average Pearson Correlation Coefficients
        """
        return self.__apcc

    @property
    def _k(self):
        """
        Array of node degrees`
        """
        return self.__k

    @property
    def _ki(self):
        """
        Nodes within-cluster degree
            (number of edges a node has with nodes in its module)
        """
        return self.__ki

    @property
    def _km(self):
        """
        Mean of within-cluster degrees
        """
        return self.__km

    @property
    def _ks(self):
        """
        Standard deviation of within-cluster degrees
        """
        return self.__ks

    @property
    def _cc(self):
        """
        Array of Clusterphobic coefficient
        """
        return self.__cc

    @property
    def _z(self):
        """
        Array of Normalized within-cluster degree
        """
        return self.__z
