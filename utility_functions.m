classdef utility_functions
    
    methods (Static)
        
        % common functions
        function [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size)
            
            if (nargin<3) || (test_size>=1)
                % If no test_size is determined, assign the test_size as
                % 0.2
                % 20% of the whole data will be used as test
                test_size = 0.2;
            end
            
            N = length(y);
            shuffle_index = randperm(N);
            X = X(shuffle_index, :);       % shuffle the data
            y = y(shuffle_index, :);       % shuffle the data
            test_N = round(N*test_size);
            train_index = 1:N-test_N;
            test_index = N-test_N+1:N;
            
            X_train = X(train_index, :);
            y_train = y(train_index, :);
            X_test = X(test_index, :);
            y_test = y(test_index, :);
            
        end
        
        function [augmented_x,augmented_y] = augment_data(x, y, augmentation_size)

            if (nargin<3)
                augmentation_size = 150e3;
            end
            
            [N,M] = size(x);
            if N<augmentation_size
                % concat necessary
                concat_time = ceil(augmentation_size/N);
                augmented_x = zeros(concat_time*N, M);
                augmented_y = zeros(concat_time*N, 1);
                for i=1:concat_time
                    start_i = (i-1)*N+1;
                    end_i = i*N;
                    shuffle_i = randperm(length(y));
                    augmented_x(start_i:end_i, :) = x(shuffle_i, :);
                    augmented_y(start_i:end_i, :) = y(shuffle_i, :);
                end
            else
                % no concat
                augmented_x = x;
                augmented_y = y;
            end

        end
        
        function ret = deriv_sigmoid_loss(z, h)
            sigmoid_loss_x = utility_functions.sigmoid_loss(z, h);
            ret = h*(1-sigmoid_loss_x)*sigmoid_loss_x;
        end

        function ret = sigmoid_loss(z,h)
            ret = 1/(1+exp(-h*z));
        end
        
    end
    
end