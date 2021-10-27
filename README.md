# T-GMRF

This repo open the python source code for "Time-varying Gaussian Markov Random Fields Learning for Multivariate Time Series Clustering".

## Using T-GMRF

T-GMRF is a novel clustering approach for Multivariate Time Series Clustering (MTS) data with particular interest in capturing its timevarying correlation patterns. It has several hyperparameters as follows:

+ **width**: the sliding windows width for GMRF series extraction.
+ **stride**: the sliding windows stride for GMRF series extraction.
+ **lamb**: the temporal smothness regularization for GMRF series extraction.
+ **beta**: the structural sparsity regularization for GMRF series extraction.
+ **diff_{threshold}**: the slope difference threshold for identifying clustering radiuses.
+ **slope_{threshold}**: the slope threshold for identifying clustring radiuses.

## Example

1. Install depedency pakcages.
```bash
conda install sklearn, snap, tslearn, tqdm, hdbscan, pyreadr

pip install 'ray[tune]'
```

2. Clone the code to local.
```bash
git clone https://github.com/Vitoom/T-GMRF.git
```

3. Run T-GMRF on BasicMotions dataset from [UEA Repo](http://www.timeseriesclassification.com/dataset.php).
```bash
python Run_Test.py
```

## Baselines
+ [**CSPCA**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/CSPCA): encoded MTS data with vectors that were dimension reduction result of covariances using PCA.
+ [**FCFW**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/FCFW): adopted a hybrid fuzzy membership matrix generated by performing DTW on all variables and reconsidering each dimension’s shapebased distance (SBD). 
+ [**HMM**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/HMM): mapped each trajectory into an HMM and then defined a suitable distance between HMMs.
+ [**MC2PCA**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/MC2PCA):It assumed a common projection axes as prototype of each cluster in K-Means clustering.

## Misc

+ [**Tune.py**](https://github.com/Vitoom/T-GMRF/blob/main/Tune.py): tune hyperparameters based on minmizing BIC or maximizing silhouette score.
+ [**generate_synthetic_data_tool.py**](https://github.com/Vitoom/T-GMRF/blob/main/Tools/generate_synthetic_data_tool.py): generate MTS dataset with various lengths, dimensions and number of instance.
+ [**Run_SyntheticData_Test.py**](https://github.com/Vitoom/T-GMRF/blob/main/Run_SyntheticData_Test.py): run T-GMRF test on sunthetic dataset.
+ [**csm.py**](https://github.com/Vitoom/T-GMRF/blob/main/Measures/csm/csm.py) compute CSM clustering measure depending on the R package, [TSclust](https://cran.r-project.org/web/packages/TSclust/index.html).
