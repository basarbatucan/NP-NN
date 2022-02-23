# A Neural Network Approach for Online Nonlinear Neyman-Pearson Classification
This is the repository for Online Nonlinear Neyman Pearson (NP) Classifier described in [1]: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9265182. 
Proposed model is an online, nonlinear NP classifier. In NP framework, the target is to maximize detection power while upper-bounding the false alarm.
Using this repository, you can test and train the model. This implementation contains hyper-parameter tuning but note that hyper-parameter spaces should be defined by the user.
Refer to [1] for detailed explanation of the hyper-parameters.


# Running the Model
NPNN code is implemented in matlab and this repo is executable by running NPNN.m

# Pipeline Parameters
Using the pipeline parameters, you can introduce a new dataset, test train ratio, target false alarm (NP framework requires target false alarm, refer to the paper for more detailed explanation).
Last parameter is related to the augmentation size. For datasets that has less number of samples, it is useful augmenting the data by shuffling and concatination to ensure convergence.
With the augmentation size, user can define a lower limit and any dataset that has less number of samples from the target will be augmented.

# Model Parameters
Existing hyper-parameter spaces are created for banana data set under ./data/. Note that they are not optimized. For performance increase, you can include additional parameters.
It is also important to note that in [1], hyper-parameter D is defined is multiplication of a constant with the dimension number of the dataset, whereas in this implementation, defined D is not multiplied by any value and directly used in Random Fourier Features generation [2].

# Evaluating and Using the results
Running the model will generate 4 different graphs. 
These graphs correspond to transient behaviour of the model during training.
In order to look at the final results, use the latest element of each array for the corresponding metric. 
Graphs of the 4 different arrays are shown below.
<img src="figures/code_output.png">

Top and bottom figures are related to train and test, respectively. The number of samples in training is related to the augmentation (explained in model parameters). 
In current case, the number of training samples is ~150k. Similarly, for test figures, there are 100 data points, where each point is an individual test of the existing 
model at different stages of the training. Please refer to the paper for more detailed explanation.

# Running the Model with a new data set
In order to run the full pipeline with a new dataset
* Make sure downloaded data has the same fields with ./data/banana.mat
* Note that ./data/banana.mat is not normalized. Any additional preprocess should be implemented.
* Make sure the downloaded data is located under the data folder.
* Update the pipeline parameter showing the directory for the input data

# Expected Decision Boundaries
When input data is 2D, it is possible to visualize decision boundaries. I included decision boundaries for banana dataset for different target false alarms. Below figure was collected from [1].
<img src="figures/decision_bound_fa_convergence.png">

Thanks!
Basarbatu Can

# References
[1] Can, Basarbatu, and Huseyin Ozkan. "A Neural Network Approach for Online Nonlinear Neyman-Pearson Classification." IEEE Access 8 (2020): 210234-210250. <br>
[2] Rahimi, Ali, and Benjamin Recht. "Random features for large-scale kernel machines." Advances in neural information processing systems 20 (2007).