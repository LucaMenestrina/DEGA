DESeq2 paper ([Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8#Sec34)) points to [http://www-huber.embl.de/DESeq2paper](http://www-huber.embl.de/DESeq2paper) for everything related to reproducibility.  
In order to retrieve the dataset on which to validate DEGA, we downloaded the file ```bottomly_sumexp.RData``` from [http://www.huber.embl.de/DESeq2paper/data/bottomly_sumexp.RData](http://www.huber.embl.de/DESeq2paper/data/bottomly_sumexp.RData), and then we extracted the files  ```bottomly_counts.csv``` and ```bottomly_phenotypes.csv```  with R:  

```R
load("bottomly_sumexp.RData")
write.csv(assay(bottomly), "bottomly_counts.csv")
write.csv(colData(bottomly), "bottomly_phenotypes.csv")
```
These source files are available here for convenience.  
<br>
This folder also contains the output files for the validation process, which are:
  File Name |  Package  | With Shrinkage
--|---|--
 ```DEGA_bottomlyResults.csv```  | DEGA | False
 ```DEGA_bottomlyWithShrinkageResults.csv```  | DEGA | True
 ```DESeq2_bottomlyResults.csv```  | DESeq2 | False
 ```DESeq2_bottomlyWithShrinkageResults.csv```  | DESeq2 | True

The comparison of the computer log2 fold changes and of the adjusted p-values (with and without shrinkage) are depicted in the plots: ```LFC.svg``` and ```FDR.svg```.  
<br>
#####References  
Love, M. I., Huber, W., Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. _Genome Biology_, _15_(12), 550. [https://doi.org/10.1186/S13059-014-0550-8/FIGURES/9](https://doi.org/10.1186/S13059-014-0550-8/FIGURES/9)  
Bottomly, D. et al. (2011). Evaluating Gene Expression in C57BL/6J and DBA/2J Mouse Striatum Using RNA-Seq and Microarrays. _PLOS ONE_, _6_(3), e17820. [https://doi.org/10.1371/JOURNAL.PONE.0017820](https://doi.org/10.1371/JOURNAL.PONE.0017820)  
