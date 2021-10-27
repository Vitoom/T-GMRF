# T-GMRF

This repo open the source code for "Time-varying Gaussian Markov Random Fields Learning for Multivariate Time Series Clustering"

T-GMRF is python source code repo for clustering Multivariate Time Series (MTS). It takes N MTS as input. It has several hyperparameters as follows:

+ **width**: the sliding windows width for GMRF series extraction.
+ **stride**: the sliding windows stride for GMRF series extraction.
+ **lamb**: the temporal smothness regularization for GMRF series extraction.
+ **beta**: the structural sparsity regularization for GMRF series extraction.
+ **diff_{threshold}**: the slope difference threshold for identifying clustering radiuses.
+ **slope_{threshold}**: the slope threshold for identifying clustring radiuses.

## Usage

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

