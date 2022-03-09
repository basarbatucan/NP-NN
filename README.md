# A Neural Network Approach for Online Nonlinear Neyman-Pearson Classification
This is the repository for Online Nonlinear Neyman Pearson (NP) Classifier described in [1]: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9265182. 
Proposed model is an online, nonlinear NP classifier. In NP framework, the target is to maximize detection power while upper-bounding the false alarm.

# Running the Code
* "main.m" is the main function for running OLNP.
* NP-NN is an NP classifier therefore it is optimized for a selected false alarm rate.
* List of different false alarms are given in "main.m". User can select any of these false alarms as target false alarm by changing the target false alarm index.
* Once the code execution is completed, training and testing phases related graphs are generated.
* Selected hyperparameters and generated outputs are saved under output directory.
* Note that there should be additional folder with the same name as your input data file for saving the output. (e.g ./data/banana.mat, ./output/banana.mat)

# Hyperparameter Tuning
* "single_experiment.m" file optimizes model for a given set of hyperparameters and single target false alarm.
* Hyperparameter space is defined in "single_experiment.m", user can increase the number of parameters for better tuning.
* Refer to [1] for detailed explanation of the hyperparameters.
* If the model is already optimized (optimum parameters are saved and available under output folder), model will not tune its parameters and continue on testing.

# Repetitive Results
* NP-NN is an online algorithm. Performance of the model is affected by the streaming order of the data. Therefore, in order to show significance of our experiements, we repeat them MC times.
* MC is available in the main function and stands for Monte Carlo repetition.
* The latest run in the main function does not contribute to the saved results. That computation is only for generating the graphs.

# Evaluating and Using the results
* Last stand alone single_experiment.m run in main.m is to generate output figures.
* There are 4 different graphs.
* These graphs correspond to transient behaviour of the model during training.
* Graphs of the 4 different arrays are shown below.
<img src="figures/code_output.png">

Top and bottom figures are related to train and test, respectively. The number of samples in training is related to the augmentation (explained in model parameters). 
In current case, the number of training samples is ~150k. Similarly, for test figures, there are 100 data points, where each point is an individual test of the existing 
model at different stages of the training. Please refer to the paper for more detailed explanation.

# Running the Model with a new data set
* Make sure downloaded data has the same fields with ./data/banana.mat
* Make sure the downloaded data is located under the data folder.
* Update the pipeline parameter showing the directory for the input data
* Include additional hyperparameters for better performance
* Create corresponding folder under output/.

# Expected Decision Boundaries
When input data is 2D, it is possible to visualize decision boundaries. I included 2 decision boundaries for target false alarms 0.05 and 0.2.<br/>
<img src="figures/db_005.png">
<img src="figures/db_020.png">

Thanks!
Basarbatu Can

# References
[1] Can, Basarbatu, and Huseyin Ozkan. "A Neural Network Approach for Online Nonlinear Neyman-Pearson Classification." IEEE Access 8 (2020): 210234-210250.