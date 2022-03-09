classdef NPNN
    
    properties
        
        % NP classification parameters
        tfpr_
        n_features_
        
        % parameters
        w_
        b_
        alpha_
        
        % hyperparameters
        eta_init_
        beta_init_
        gamma_
        sigmoid_h_
        lambda_
        D_
        g_
        
        % results
        tpr_train_array_
        fpr_train_array_
        tpr_test_array_
        fpr_test_array_
        neg_class_weight_train_array_
        pos_class_weight_train_array_
        
    end
    
    methods
        
        function obj = NPNN(eta_init, beta_init, gamma, sigmoid_h, lambda, D, g, n_features, tfpr)
            
            % init hyperparameters
            obj.eta_init_ = eta_init;
            obj.beta_init_ = beta_init;
            obj.gamma_ = gamma;
            obj.sigmoid_h_ = sigmoid_h;
            obj.lambda_ = lambda;
            obj.D_ = D;
            obj.g_ = g;
            
            % init model parameters
            obj.n_features_ = n_features;
            obj.tfpr_ = tfpr;
            
        end
        
        function obj = train(obj, X_train, y_train, X_test, y_test, test_repeat)
            
            % init NP classification parameters
            tfpr = obj.tfpr_;
            n_features = obj.n_features_;
            
            % init hyperparameters from constructor
            eta_init = obj.eta_init_;
            beta_init = obj.beta_init_;
            gamma = obj.gamma_;
            sigmoid_h = obj.sigmoid_h_;
            lambda = obj.lambda_;
            D = obj.D_;
            g = obj.g_;
            
            % init training parameters
            n_samples_train = size(X_train, 1);
            n_samples_test = size(X_test, 1);
            number_of_negative_samples = sum(y_train==-1);
            alpha = mvnrnd(zeros(n_features,1), 2*g*eye(n_features), D)'; % random fourier features
            w = randn(2*D, 1)*1e-4;
            b = randn*1e-4;
            eta = eta_init;
            beta_init = beta_init/number_of_negative_samples; % after augmentation, we scale learning rate for class specific weight
            beta = beta_init;
            negative_sample_buffer_size = max(round(2/tfpr), 200);
            negative_sample_buffer = zeros(1, negative_sample_buffer_size);
            negative_sample_buffer_index = 1;
            
            % save test related parameters for online evaluation
            index_act_pos = y_test == 1;
            N_act_pos = sum(index_act_pos);
            index_act_neg = y_test == -1;
            N_act_neg = sum(index_act_neg);

            % init accumulators
            tp = 0;
            fp = 0;
            test_i = linspace(1, n_samples_train, test_repeat+1);
            test_i=round(test_i(2:end));
            current_test_i = 1;

            % array outputs
            tpr_test_array = zeros(1, test_repeat);
            fpr_test_array = zeros(1, test_repeat);
            tpr_train_array = zeros(1, n_samples_train);
            fpr_train_array = zeros(1, n_samples_train);
            neg_class_weight_train_array = zeros(1, n_samples_train);
            pos_class_weight_train_array = zeros(1, n_samples_train);
            gamma_array = zeros(1, n_samples_train);
            number_of_positive_samples = 1;
            number_of_negative_samples = 1;

            %add initials
            neg_class_weight_train_array(1) = 2*gamma;
            pos_class_weight_train_array(1) = 2*gamma;

            % online training
            for i=1:n_samples_train

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
                        xt_r_tmp = xt_tmp*alpha;
                        xt_projected_tmp = (1/sqrt(D))*[cos(xt_r_tmp), sin(xt_r_tmp)];
                        y_discriminant_tmp = xt_projected_tmp*w+b;
                        y_predict_tmp(j) = sign(y_discriminant_tmp);
                    end

                    % evaluate
                    index_pred_pos = y_predict_tmp == 1;

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
            
            % save calculated parameters
            obj.w_ = w;
            obj.b_ = b;
            obj.alpha_ = alpha;
            
            % save the results
            obj.tpr_train_array_ = tpr_train_array;
            obj.fpr_train_array_ = fpr_train_array;
            obj.tpr_test_array_ = tpr_test_array;
            obj.fpr_test_array_ = fpr_test_array;
            obj.neg_class_weight_train_array_ = neg_class_weight_train_array;
            obj.pos_class_weight_train_array_ = pos_class_weight_train_array;
            
        end
        
        function plot_results(obj)
            
            subplot(2,2,1)
            plot(obj.tpr_train_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Training Samples');
            ylabel('Train TPR');

            subplot(2,2,2)
            plot(obj.fpr_train_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Training Samples');
            ylabel('Train FPR');

            subplot(2,2,3)
            plot(obj.tpr_test_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Tests');
            ylabel('Test TPR');

            subplot(2,2,4)
            plot(obj.fpr_test_array_, 'LineWidth', 2);grid on;
            xlabel('Number of Tests');
            ylabel('Test FPR');
            
        end
        
    end
    
end