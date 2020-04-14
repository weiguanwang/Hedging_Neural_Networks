# README	

## Contact and suggested citation

Author:

- Prof. Johannes Ruf [j.ruf@lse.ac.uk](), http://www.maths.lse.ac.uk/Personal/jruf/
- Weiguan Wang [w.wang34@lse.ac.uk]()

Address: 

Department of Mathematics, London School of Economics and Political Science, London, United Kingdom

Date: 14 April 2020



Suggested citation:

J. Ruf and W. Wang, Hedging with Neural Networks, arxiv , 2020

This paper can be found at **fill**.

## Introduction

This documentation explains the structures of the code and data folders to reproduce the results in the paper Ruf and Wang 2020. To run the code provided here, the user needs to:

1. Overwrite the `DATA_DIR` variable in the `setup.py` to your own choice.
2. Obtain raw data (should you want to work with real datasets) and rename files as detailed in [link](#Data folder structure).

## Code structure

The code consists of four sub folders. They are `libaray`, `Simulation`, `OptionMetrics` and `Euroxx`. The `library` implement functions used by other parts of the code. It consists of:

1. `bs.py` : This file contains a function used to simulate of the Black-Scholes dataset.
2. `heston.py`: This file contains functions used to simulate the Heston model, as well as calculating option prices.
3. `vix.py`: This file contains the function that simulates an Ornstein-Uhlenbeck  process, used as the fake VIX.
4. `simulation.py`: This file contains functions that implement the CBOE rules, and organize data.
5. `stoxx.py`: This file contains function used to clean the Euro Stoxx 50 dataset only. 
6. `common.py` : This file contains functions that calculate and inspect hedging error, and run permutations.
7. `network.py` : This file implements the HedgeNet and its auxiliary functions concerning training and evaluation.
8. `cleaner_aux.py`: This file contains functions used to clean raw data and generate clean data.
9. `regression_aux.py`: This file contains functions that implement the linear regression methods.
10. `loader_aux.py` : This file contains functions used to load data before running ANN or linear regressions.
11. `plot.py`: This file contains functions used to plot diagnostic figures. 



In each of the other three folder, there are two python files that are used by other notebooks:

1. `setup.py`: This file contains all the flags to configure experiments. It varies by datasets, and contains two major functions:
   1. It specifies hedging period, time window size, data cleaning choice, and other experiment setup.
   2. It specifies the location of raw data, clean data  and result.
2.  `Load_Clean_aux.py` loads the clean data and implements some extra cleaning, before running linear regression or ANNs.

The notebooks have a very similar structure as follows:

1. In the simulation folder, the first notebook implements the simulation of data, for Black-Scholes and Heston. In the Option Metrics or Euroxx folder, the first notebook implements the cleaning of real raw datasets downloaded from data providers. 
2. `2_Regression_Generate.ipynb` implements all linear regressions on sensitivities and stores the PNL files.
3. `3_Tuning _Hyper.ipynb` implements the tuning of $L^2 $ regularisation parameters. 
4. `4_Network.ipynb` implements the training of the ANN and store the PNL files.
6. `5_Diagnostic.ipynb` makes the table that summarize PNL files  in terms of given performance measure, across several experimental setups, i.e. globally.
7. `6_Local_Diag_And_Plots.ipynb` implements the diagnostics of PNL files for a single experimental setup. Plots made from PNL files are generated in this file. They includes linear regression coefficients, mean squared hedging error plots, PNL vs sensitivities, and etc.
8. `7_Analysis_of_(Semi-)CleanData.ipynb` implements the analysis of raw or clean data. They include histogram of certain features, number of samples in each time window, volatility, leverage effect and so on.
9. `8_Permute_VIX_Analysis.ipynb` implements the analysis of permutation and fake VIX experiments. The implementation of the experiment is done in notebook 4 and 5, by giving the corresponding setup flags. This notebook only exists for Simulation and OptionMetrics folders.

## Data folder structure

Before running the code, one needs to specify the directory that stores the simulation data, (or real data) and the results. This is done by overwriting the `DATA_DIR` variable in each of the `setup.py` file. 

The data folders  have two common subfolders,

1. `CleanData`: It stores simulation data in case of Black-Scholes or Heston data, or clean data after raw data is cleaned by `1_Clean.ipynb` in case of real data.
2. `Result`: It store all the PNL files and other auxiliary, either from the linear regressions or ANN. They also include  tables made by `5_Diagnostic.ipynb`. For the ANN, it additionally contains loss plots, checkpoints and others. For the linear regression, it additionally contains regression coefficients, standard error and etc.

For the two real datasets, there is an extra folder `RawData` to store to data given by data providers. Data needs to be arranged and renamed in the following way for the code to run.

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

