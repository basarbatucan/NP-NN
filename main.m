close
clear
clc

% only look at first 5 for other analysis
tfprs = [5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
tfpr_index = 5;
MC = 32;
forced_parameter_tuning_flag = 1;

% data_name = 'avila';
% data_name = 'banana';
% data_name = 'covertype';
% data_name = 'fourclass';
% data_name = 'miniboone_pid';
% data_name = 'phishing';
% data_name = 'satellite';
data_name = 'telescope';

out_data = sprintf('./output/%s/res_%03d.mat', data_name, tfpr_index);
out_hyper = sprintf('./output/%s/res_hyper_%03d.mat', data_name, tfpr_index);

tpr_train_array_all = cell(1, MC);
fpr_train_array_all = cell(1, MC);
neg_class_weight_train_array_all = cell(1, MC);
pos_class_weight_train_array_all = cell(1, MC);
tpr_test_array_all = cell(1, MC);
fpr_test_array_all = cell(1, MC);

% cross validation part
tfpr = tfprs(tfpr_index);
if (~isfile(out_hyper)) || (forced_parameter_tuning_flag == 1)
    disp('Running parameter tuning...');
    % no hyperparameters available, run
    test_repeat = 1;
    model = single_experiment(tfpr, data_name, test_repeat, []);
    % get the hyperparameters
    eta_init = model.eta_init_;
    beta_init = model.beta_init_;
    gamma = model.gamma_;
    sigmoid_h = model.sigmoid_h_;
    lambda = model.lambda_;
    D = model.D_;
    g = model.g_;
    % save hyper-parameters
    save(out_hyper, 'eta_init', 'beta_init', 'gamma', 'sigmoid_h', 'lambda', 'D', 'g');
end

% hyperparameter is available
hyper_params = load(out_hyper);
% run the model with hyperparams
test_repeat = 100;

% test tun with the selected parameters
% comment this part in order to ignore one additional
disp('Starting single run...');
t_start = tic;
model = single_experiment(tfpr, data_name, test_repeat, hyper_params);
t_end = toc(t_start);
disp(['test run completed in ',num2str(t_end)]);
pause;

% run MCs
test_tstart = tic;
parfor i=1:MC
    % run the model
    model = single_experiment(tfpr, data_name, test_repeat, hyper_params);
    % save the results
    tpr_train_array_all{i} = model.tpr_train_array_;
    fpr_train_array_all{i} = model.fpr_train_array_;
    neg_class_weight_train_array_all{i} = model.neg_class_weight_train_array_;
    pos_class_weight_train_array_all{i} = model.pos_class_weight_train_array_;
    tpr_test_array_all{i} = model.tpr_test_array_;
    fpr_test_array_all{i} = model.fpr_test_array_;
end
test_indices = model.test_indices_;
test_tend = toc(test_tstart);
fprintf('Time elapsed for testing with %d test_repeat: %.3f\n', test_repeat, test_tend);

save(sprintf('./output/%s/res_%03d',data_name, tfpr_index),...
    'tpr_train_array_all',...
    'fpr_train_array_all',...
    'neg_class_weight_train_array_all',...
    'pos_class_weight_train_array_all',...
    'tpr_test_array_all',...
    'fpr_test_array_all',...
    'test_indices');

% run 1 last time for for generating the output figures
% note that the results generated at this point is not saved, this is only
% for generating decision boundaries and transient outputs
% model = single_experiment(tfpr, data_name, test_repeat, hyper_params);