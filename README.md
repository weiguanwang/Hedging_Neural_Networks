# README	



## Contact and citation

Weiguan Wang [w.wang34@lse.ac.uk]()

Prof. Johannes Ruf [j.ruf@lse.ac.uk](), http://www.maths.lse.ac.uk/Personal/jruf/

Department of Mathematics, London School of Economics and Political Science, London, United Kingdom

14 April 2020



Suggested citation:

J. Ruf and W. Wang, Hedging with Neural Networks, SSRN 3580132, 2020. Follow the **[link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3580132)** to download.

## Introduction

This documentation explains the code structure and data folders to reproduce the results in Ruf and Wang (2020). To run the code provided here, the user needs to:

1. Overwrite the `DATA_DIR` variable in the `setup.py` to your own choice.
2. Obtain raw data (should you want to work with real datasets) and rename files as detailed in [link](#Data folder structure).

## Code structure

The code consists of four subfolders. They are `libaray`, `Simulation`, `OptionMetrics`, and `Euroxx`. The `library` folder contains functions used by other parts of the code. The `library` consists of:

1. `bs.py` : This file contains a function used to simulate the Black-Scholes dataset.
2. `cleaner_aux.py`: This file contains functions used to clean raw data.
3. `common.py` : This file contains functions that calculate and inspect the hedging error.
4. `heston.py`: This file contains functions used to simulate the Heston dataset, as well as calculating option prices in the Heston model.
5. `loader_aux.py` : This file contains functions used to load clean data (before training the ANN or linear regressions).
6. `network.py` : This file implements HedgeNet and auxiliary functions.
7. `plot.py`: This file contains functions used to plot diagnostic figures. 
8. `regression_aux.py`: This file contains functions that implement the linear regression methods.
9. `simulation.py`: This file contains functions that implement the CBOE rules, and organize data.
10. `stoxx.py`: This file contains function used to clean the Euro Stoxx 50 dataset only. 
11. `vix.py`: This file contains the function that simulates an Ornstein-Uhlenbeck  process, used as the fake VIX feature.



In each of the other three folder, there are two python files that are used by other notebooks:

1. `Setup.py`: This file contains all the flags to configure experiments. It varies by datasets, and contains two major configurations:
   1. It specifies the hedging period, time window size, data cleaning choice, and other experimental setup.
   2. It specifies the location of raw data, clean data, and the stored results.
2.  `Load_Clean_aux.py` loads the clean data and implements some extra cleaning, before running linear regressions or ANNs.

The notebooks have a very similar structure as follows:

1. In the simulation folder, the first notebook implements the data simulation. In the OptionMetrics and Euroxx folder, the first notebook implements the cleaning of the real raw datasets downloaded from data providers. 
2. `2_Regression_Generate.ipynb` implements all linear regressions on sensitivities and stores the PNL (MSHE) files.
3. `3_Tuning _Hyper.ipynb` implements the tuning of $L^2 $ regularisation parameters. 
4. `4_Network.ipynb` implements the training of the ANN and stores the PNL files (MSHE of ANN).
6. `5_Diagnostic.ipynb` creates tables to summarize PNL (MSHE) files in terms of given performance measure, across several experimental setups, i.e. globally for each dataset.
7. `6_Local_Diag_And_Plots.ipynb` implements the diagnostics of PNL files for a single experimental setup. Plots made from PNL files are generated in this file. They include linear regression coefficients, mean squared hedging error plots, MSHE vs sensitivities, etc.
8. `7_Analysis_of_(Semi-)CleanData.ipynb` implements the analysis of raw and clean data. They include histograms of certain features, number of samples in each time window, volatility, leverage effect, etc.
9. `8_Permute_VIX_Analysis.ipynb` implements the analysis of permutation and fake VIX experiments. The implementation of the experiment is done in notebook 4 and 5, by giving the corresponding setup flags. This notebook only exists for the Simulation and OptionMetrics folders.

## Data folder structure

Before running the code, one needs to specify the directory that stores the simulation data, (or real data) and the results. This is done by overwriting the `DATA_DIR` variable in each of the `setup.py` file. 

The data folders  have two common subfolders,

1. `CleanData`: It stores simulation data in case of Black-Scholes or Heston data, or clean data genereated by `1_Clean.ipynb` in case of real data.
2. `Result`: It store the PNL files and other auxiliary files, either from the linear regressions or ANN. They also include  tables made by `5_Diagnostic.ipynb`. For the ANN, it additionally contains loss plots, checkpoints, etc. For the linear regression, it additionally contains regression coefficients, standard errors, etc.

For the two real datasets, there is an extra folder `RawData` to store data given by data providers. Data needs to be arranged and renamed in the following way for the code to run.

- For the S\&P 500 data. There are 4 files:

  1. `option_price.csv` contains option quotes downloaded from OptionMetrics. 

  2. `spx500.csv` contains the close-of-day price of S\&P 500. 

  3. `onr.csv` contains the overnight LIBOR rate downloaded from Bloomberg.

  4. `interest_rate.csv` contains the interest rate derived from zero-coupon bond for maturity larger than 7 days, downloaded from OptionMetrics.

- For the Euro Stoxx data. Data needs to be put in four folders:

  1. `futures` contains two files, `futures.csv` and `refData.csv`; the former contains the tick trading data of futures, and the latter contains the contract specifications of futures in the former. 
  2. `options` contains two files, `options.csv` and `refData.csv`; they are tick trading data of options and their reference.
  3. `interest_rate` contains seven files. They are `LIBOR_EURO_ON`, `LIBOR_EURO_1M.csv`,  `LIBOR_EURO_3M.csv`, `LIBOR_EURO_6M.csv` `LIBOR_EURO_12M.csv`; namely, LIBOR rate of overnight, maturity 1 month, 3 months, 6 months, 12 months. The other two files are `ZERO_EURO_5Y` and `ZERO_EURO_10Y`; namely, interest rate derived from zero-coupon bond of maturity 5  and 10 years.
  4. `stoxx50.csv` is the end-of-day spot of Euro Stoxx 50 index.
  



## Package information

| Package      | Version |
| ------------ | ------- |
| Anaconda     | 2019.03 |
| Keras        | 2.2.4   |
| Python       | 3.6     |
| Numpy        | 1.16.3  |
| Pandas       | 0.24.2  |
| Scikit-learn | 0.20.3  |
| Scipy        | 1.2.1   |
| Seaborn      | 0.9     |
| Tensorflow   | 1.13.1  |

