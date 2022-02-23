clear
close all
clc

% Define pipeline variables
input_data_dir = './data/banana.mat';
val_size = 0.15;
test_size = 0.15;
augmentation_size = 150e3;
test_repeat = 100;
cross_val_MC = 8;
tfpr = 0.2;

% Define model hyper-parameter space
hyperparams.eta_init = 0.01;
hyperparams.beta_init = 100;
hyperparams.gamma = 1;
hyperparams.sigmoid_h = -1;
hyperparams.lambda = 0;
hyperparams.D = [2,5,10];
hyperparams.g = [0.01,1,10];

% generate hyper-parameter space 
hyperparam_space = utility_functions.generate_hyperparameter_space_NPNN(hyperparams);
hyperparam_number = length(hyperparam_space);
cross_val_scores = zeros(cross_val_MC, hyperparam_number);

% Read Data
data = load(input_data_dir);
[X_train, X_val, X_test, y_train, y_val, y_test] = utility_functions.train_val_test_split(data.x, data.y, val_size, test_size);
n_features = size(X_train, 2);

% cross validation
if hyperparam_number>1
    
    % force hyperparameter tuning
    X_train_ = X_train;
    y_train_ = y_train;
    X_val_ = X_val;
    y_val_ = y_val;
    
    % normalization
    [X_train_, mu_train, sigma_train] = zscore(X_train_);
    for i=1:n_features
        X_val_(:,i) = (X_val_(:,i)-mu_train(i))/sigma_train(i);
    end
    
    % compare cross validations
    for i=1:length(hyperparam_space)
        parfor j=1:cross_val_MC
            
            eta_init = hyperparam_space{i}.eta_init;
            beta_init = hyperparam_space{i}.beta_init;
            gamma = hyperparam_space{i}.gamma;
            sigmoid_h = hyperparam_space{i}.sigmoid_h;
            lambda = hyperparam_space{i}.lambda;
            D = hyperparam_space{i}.D;
            g = hyperparam_space{i}.g;
            
            % load the model
            model = NPNN(eta_init, beta_init, gamma, sigmoid_h, lambda, D, g, n_features, tfpr);
            
            % augmentation (also includes shuffling)
            [X_train__, y_train__] = utility_functions.augment_data(X_train_, y_train_, augmentation_size);
            
            % train the model
            model = model.train(X_train__, y_train__, X_val_, y_val_, 1);

            % evaluate NP score
            tpr = model.tpr_test_array_(end);
            fpr = model.fpr_test_array_(end);
            NP_score = utility_functions.get_NP_score(tpr, fpr, tfpr);
            cross_val_scores(j,i) = NP_score;
            
        end
    end
    
    % make decision based on mean of the NP scores
    cross_val_scores_ = mean(cross_val_scores);
    
    % find out the best hyperparameter set
    % for NP score, lesser is better
    [~, target_hyperparameter_index] = min(cross_val_scores_);
    
    % select optimum hyperparameters
    eta_init = hyperparam_space{target_hyperparameter_index}.eta_init;
    beta_init = hyperparam_space{target_hyperparameter_index}.beta_init;
    gamma = hyperparam_space{target_hyperparameter_index}.gamma;
    sigmoid_h = hyperparam_space{target_hyperparameter_index}.sigmoid_h;
    lambda = hyperparam_space{target_hyperparameter_index}.lambda;
    D = hyperparam_space{target_hyperparameter_index}.D;
    g = hyperparam_space{target_hyperparameter_index}.g;
    
else
    
    % there is only one hyperparameter defined
    eta_init = hyperparam_space{1}.eta_init;
    beta_init = hyperparam_space{1}.beta_init;
    gamma = hyperparam_space{1}.gamma;
    sigmoid_h = hyperparam_space{1}.sigmoid_h;
    lambda = hyperparam_space{1}.lambda;
    D = hyperparam_space{1}.D;
    g = hyperparam_space{1}.g;
    
end

%% training
% since hyperparameter tuning is completed, merge train and val
X_train = [X_train;X_val];
y_train = [y_train;y_val];
[X_train, mu_train, sigma_train] = zscore(X_train);
for i=1:n_features
    X_test(:,i) = (X_test(:,i)-mu_train(i))/sigma_train(i);
end

% Preprocessing
[X_train, y_train] = utility_functions.augment_data(X_train, y_train, augmentation_size);

% load the model
model = NPNN(eta_init, beta_init, gamma, sigmoid_h, lambda, D, g, n_features, tfpr);

% train the model
model = model.train(X_train, y_train, X_test, y_test, 100);

% plot the results
model.plot_results();

% plot decision boundaries
if n_features == 2
    utility_functions.plot_decision_boundary(model, X_test, y_test)
end