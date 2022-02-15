clear
close all
clc

% Define pipeline variables
input_data_dir = './data/banana.mat';
test_size = 0.2;
augmentation_size = 150e3;
test_repeat = 100;

% Define model parameters
tfpr = 0.05;
eta_init = 0.01;                          % perceptron learning rate
beta_init = 100;                          % class weight initial learning rate
gamma = 1;
sigmoid_h = -1;
lambda = 0;
D_multiplier = 10;
g_multiplier = 1;

% Read Data
data = load(input_data_dir);
[X_train, X_test, y_train, y_test] = utility_functions.train_test_split(data.x, data.y, test_size);

% Save test set properties for evaluation
index_act_pos = y_test == 1;
N_act_pos = sum(index_act_pos);
index_act_neg = y_test == -1;
N_act_neg = sum(index_act_neg);

% normalize with respect to train
[X_train, mu_train, sigma_train] = zscore(X_train);

% Preprocessing
[X_train, y_train] = utility_functions.augment_data(X_train, y_train, augmentation_size);

% Initialize model parameters
n_samples = size(X_train, 1);
n_samples_test = size(X_test, 1);
n_features = size(X_train, 2);
number_of_negative_samples = sum(y_train==-1);
D = n_features*D_multiplier;
g = 1/n_features*g_multiplier;
w = randn(2*D, 1)*1e-4;
b = randn*1e-4;
eta = eta_init;
beta_init = beta_init/number_of_negative_samples; % after augmentation, we scale learning rate for class specific weight
beta = beta_init;
alpha = mvnrnd(zeros(n_features,1), 2*g*eye(n_features), D)'; % random fourier features
negative_sample_buffer_size = max(round(2/tfpr), 200);
negative_sample_buffer = zeros(1, negative_sample_buffer_size);
negative_sample_buffer_index = 1;

% normalize test with trains mean and std
for i=1:n_features
    X_test(:,i) = (X_test(:,i)-mu_train(i))/sigma_train(i);
end

% initialize accumulators
tp = 0;
fp = 0;
test_i = linspace(1, n_samples, test_repeat+1);
test_i=round(test_i(2:end));
current_test_i = 1;

% array outputs
tpr_test_array = zeros(1, test_repeat);
fpr_test_array = zeros(1, test_repeat);
tpr_train_array = zeros(1, n_samples);
fpr_train_array = zeros(1, n_samples);
neg_class_weight_train_array = zeros(1, n_samples);
pos_class_weight_train_array = zeros(1, n_samples);
gamma_array = zeros(1, n_samples);
number_of_positive_samples = 1;
number_of_negative_samples = 1;

%add initials
neg_class_weight_train_array(1) = 2*gamma;
pos_class_weight_train_array(1) = 2*gamma;
            
% online training
for i=1:n_samples
    
    % take the input data
    xt = X_train(i, :);
    yt = y_train(i, :);
    
    % project input to the higher dimensional input space
    xt_r = xt*alpha;
    xt_projected = (1/sqrt(D))*[cos(xt_r), sin(xt_r)];
    
    % make prediction in the higher dimensional sppace
    y_discriminant = xt_projected*w+b; 
    yt_predict = sign(y_discriminant);
    
    % save tp and fp
    if yt == 1
        if yt_predict == 1
            tp = tp+1;
        end
    else
        if yt_predict == 1
            fp = fp+1;
        end
    end
    tpr_train_array(i) = tp/number_of_positive_samples;
    fpr_train_array(i) = fp/number_of_negative_samples;
    
    % save gamma
    gamma_array(i) = gamma;
    
    % test case
    if i==test_i(current_test_i)
        
        % run the prediction for test case
        y_predict_tmp = zeros(n_samples_test,1);
        for j=1:n_samples_test
            xt_tmp = X_test(j,:);
            yt_tmp = y_test(j,:);
            xt_r_tmp = xt_tmp*alpha;
            xt_projected_tmp = (1/sqrt(D))*[cos(xt_r_tmp), sin(xt_r_tmp)];
            y_discriminant_tmp = xt_projected_tmp*w+b;
            y_predict_tmp(j) = sign(y_discriminant_tmp);
        end
        
        % evaluate
        index_pred_pos = y_predict_tmp == 1;
        N_pred_pos = sum(index_pred_pos);
        index_pred_neg = y_predict_tmp == -1;
        N_pred_neg = sum(index_pred_neg);
        
        tp_tmp = sum(index_pred_pos & index_act_pos);
        tpr_tmp = tp_tmp/N_act_pos;
        tpr_test_array(current_test_i) = tpr_tmp;
        
        fp_tmp = sum(index_pred_pos & index_act_neg);
        fpr_tmp = fp_tmp/N_act_neg;
        fpr_test_array(current_test_i) = fpr_tmp;
        
        current_test_i=current_test_i+1;
        
    end
    
    % update the buffer with the current prediction
    if yt == -1

        % modify the size of the FPR estimation buffer
        if negative_sample_buffer_index == negative_sample_buffer_size
            negative_sample_buffer(1:end-1) = negative_sample_buffer(2:end);
        else
            negative_sample_buffer_index = negative_sample_buffer_index + 1;
        end

        if yt_predict == 1
            % false positive
            negative_sample_buffer(negative_sample_buffer_index) = 1;
        else
            % true negative
            negative_sample_buffer(negative_sample_buffer_index) = 0;
        end

    end
    
    % estimate the FPR of the current model using the moving buffer
    if negative_sample_buffer_index == negative_sample_buffer_size
        estimated_FPR = mean(negative_sample_buffer);
    else
        estimated_FPR = mean(negative_sample_buffer(1:negative_sample_buffer_index));
    end
    
    % Calculate instant loss
    z = yt*y_discriminant;
    dloss_dz = utility_functions.deriv_sigmoid_loss(z,sigmoid_h);
    dz_dw = yt*xt_projected';
    dz_db = yt;
    dloss_dw = dloss_dz*dz_dw;
    dloss_db = dloss_dz*dz_db;
    
    % y(t), calculate mu(t)
    if yt==1
        % mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
        mu = (number_of_positive_samples + number_of_negative_samples)/number_of_positive_samples;
        % save class costs
        pos_class_weight_train_array(i) = mu;
        if i>1
            neg_class_weight_train_array(i) = neg_class_weight_train_array(i-1);
        end
    else
        % mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
        mu = gamma*(number_of_positive_samples + number_of_negative_samples)/number_of_negative_samples;
        % save class costs
        neg_class_weight_train_array(i) = mu;
        if i>1
            pos_class_weight_train_array(i) = pos_class_weight_train_array(i-1);
        end
    end
    
    % SGD
    
    % update w and b
    w = (1-eta*lambda)*w-eta*mu*dloss_dw;
    b = b-eta*mu*dloss_db;
    
    % update projection
    result_der = (1/sqrt(D))*(-w(1:D,:)'.*sin(xt_r) + w(D+1:end,:)'.*cos(xt_r));
    alpha = alpha - eta*mu*utility_functions.deriv_sigmoid_loss(z,sigmoid_h)*yt*xt'*result_der;
    
    % update learning rate of perceptron
    eta = eta_init/(1+lambda*(number_of_positive_samples + number_of_negative_samples));
    
    % y(t)
    if yt==1
        % calculate n_plus(t)
        number_of_positive_samples = number_of_positive_samples + 1;
    else
        % calculate n_minus(t)
        number_of_negative_samples = number_of_negative_samples + 1;
        % calculate gamma(t)
        gamma = gamma*(1+beta*(estimated_FPR - tfpr));
    end

    % update uzawa gain
    beta = beta_init/(1+lambda*(number_of_positive_samples + number_of_negative_samples));
    
end

% plot the decision boundary if the data is 2D
subplot(2,2,1)
plot(tpr_train_array, 'LineWidth', 2);grid on;
xlabel('Number of Training Samples');
ylabel('Train TPR');

subplot(2,2,2)
plot(fpr_train_array, 'LineWidth', 2);grid on;
xlabel('Number of Training Samples');
ylabel('Train FPR');

subplot(2,2,3)
plot(tpr_test_array, 'LineWidth', 2);grid on;
xlabel('Number of Tests');
ylabel('Test TPR');

subplot(2,2,4)
plot(fpr_test_array, 'LineWidth', 2);grid on;
xlabel('Number of Tests');
ylabel('Test FPR');

% plot the decision boundaries when 2D
if n_features == 2
    utility_functions.plot_decision_boundary(alpha, D, w, b, X_test, y_test);
end