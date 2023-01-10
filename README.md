<div align="center">
  <img src="https://raw.githubusercontent.com/LucaMenestrina/DEGA/main/logo.svg", alt="DEGA", width=60%>
</div>

<div align="center">
  <h2>
    DEGA: A Python Package for Differential Gene Expression Analysis
  </h2>
</div>

**Differential gene expression analysis** is an important tool for identifying genes that display a significantly altered expression in response to specific stimuli.  
**DEGA** is a Python package for differential expression analysis. It is an implementation of the core algorithm of the [R package DESeq2](https://bioconductor.org/packages/DESeq2/). Along with the differential testing algorithm, DEGA also provides high-level functions for **dataset exploration** and **results interpretation**.

#### Installation
```python
pip install DEGA
```

#### Quick Start
```python
import DEGA

dega = DEGA.dataset(countsData, phenotypeData, designFormula="factor")
dega.analyse()
```
For a complete use case check the [Jupyter Notebook](https://github.com/LucaMenestrina/DEGA/blob/main/validation/DEGA.ipynb)

<!--
#### Citation Note
Please cite [our paper](url) if you use *DEGA* in your own work:

```
@article {TAG,
         title = {DEGA: a Python package for differential gene expression analysis},
         author = {Menestrina, Luca and Recanatini, Maurizio},
         journal = {Journal},
         volume = {Vol},
         year = {Year},
         doi = {doi},
         URL = {url},
         publisher = {Publisher},
}
```
-->
