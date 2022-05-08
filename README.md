# T-GMRF

This repo open the sources code for "Time-varying Gaussian Markov Random Fields Learning for Multivariate Time Series Clustering".

## Using T-GMRF

T-GMRF is a novel clustering approach for Multivariate Time Series Clustering (MTS) data with particular interest in capturing its timevarying correlation patterns. We introduce a time-varying Gaussian Markov random field (T-GMRF) model to desciribe the correlation structure between MTS variables. The major classes in this repo include:

+ [**TGMRF.py**](https://github.com/Vitoom/T-GMRF/blob/main/TGMRF.py): T-GMRF learning class to extract GMRF series. It depends on Random Block Descent and ADMM solvers [here](https://github.com/Vitoom/T-GMRF/tree/main/Solver).
+ [**MD_Cluster.py**](https://github.com/Vitoom/T-GMRF/blob/main/MD_Cluster.py): Multi-density based clustering class using fast density estimation procedure.

It has several hyperparameters as follows:

+ **width**: The sliding windows width for GMRF series extraction.
+ **stride**: The sliding windows stride for GMRF series extraction.
+ **lamb**: The temporal smothness regularization for GMRF series extraction.
+ **beta**: The structural sparsity regularization for GMRF series extraction.
+ **diff_{threshold}**: The slope difference threshold for identifying clustering radiuses.
+ **slope_{threshold}**: The slope threshold for identifying clustring radiuses.

## Example

1. Install depedency pakcages.
```bash
pip install sklearn, snap, tslearn, tqdm, hdbscan, pyreadr, 'ray[tune]'
```

2. Clone the code to local.
```bash
git clone https://github.com/Vitoom/T-GMRF.git
```

3. Run quick test of T-GMRF on a small dataset, BasicMotions, opened from [UEA Repo](http://www.timeseriesclassification.com/dataset.php).
```bash
cd T-GMRF/
python Run_Test.py
```

## Baselines

Open partially source codes of the compared baselines.

+ [**FCFW**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/FCFW): KMeans Clustering on a hybrid fuzzy membership matrix generated by performing DTW on all variables and reconsidering each dimension’s shapebased distance (SBD). 
+ [**CSPCA**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/CSPCA): Encode MTS data with vectors that were dimension reduction result of covariances using PCA and cluster them with KMeans.
+ [**HMM**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/HMM): Map each trajectory into an HMM and then defined a suitable distance between HMMs and assign clustering label with  KMeans.
+ [**MC2PCA**](https://github.com/Vitoom/T-GMRF/tree/main/Baselines/MC2PCA): Adopt a common projection axes as prototype of each cluster in K-Means clustering.

## Misc

+ [**RI.py**](https://github.com/Vitoom/T-GMRF/blob/main/Measures/RI.py): A metric measuring the agreement between two partitions and shows how much the clustering results are close to the ground truth.
+ [**CSM.py**](https://github.com/Vitoom/T-GMRF/blob/main/Measures/CSM.py): A metric measuring cluster similarity between the true cluster label and the clustering result.
+ [**Tune.py**](https://github.com/Vitoom/T-GMRF/blob/main/Tune.py): Tune hyperparameters based on minmizing BIC or maximizing silhouette score.
+ [**generate_synthetic_data_tool.py**](https://github.com/Vitoom/T-GMRF/blob/main/Tools/generate_synthetic_data_tool.py): Generate MTS dataset with various lengths, dimensions and number of instance using prescribed GRMF series.
+ [**Run_SyntheticData_Test.py**](https://github.com/Vitoom/T-GMRF/blob/main/Run_SyntheticData_Test.py): Run T-GMRF test on synthetic datasets.
