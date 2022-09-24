#!/usr/bin/env python

__author__ = "Basarbatu Can, Huseyin Ozkan"
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Basarbatu Can"
__status__ = "Dev"

import numpy as np
import warnings
from sklearn.metrics import confusion_matrix

class npnn:

    def __init__(self, tfpr=0.1, eta_init=0.01, beta_init=100, sigmoid_h=-1, Lambda=0, D=2, g=0.1):

        # hyperparameters
        self.tfpr = tfpr
        self.eta_init = eta_init
        self.beta_init = beta_init
        self.sigmoid_h = sigmoid_h
        self.Lambda = Lambda
        self.D = D
        self.g = g
    
        # parameters (these features are assigned after training is completed)
        self.w_ = None # perceptron weight
        self.b_ = None # perceptron bias
        self.alpha_ = None # Fourier projection weights

        # below arrays are used to store learning performance of npnn
        self.tpr_train_array_ = None
        self.fpr_train_array_ = None
        self.neg_class_weight_train_array_ = None # learned weight for negative class (note that we assume binary classes are 1 and -1)
        self.pos_class_weight_train_array_ = None # learned weight for positive class

    #def fit(self, X, y, n_samples_augmented_min=150e3):
    def fit(self, X, y, **fit_params):

        # take the parameters
        tfpr = self.tfpr
        eta_init = self.eta_init
        beta_init = self.beta_init
        gamma = 1 # this is initialized as 1, can change later
        sigmoid_h = self.sigmoid_h
        Lambda = self.Lambda
        D = self.D
        g = self.g

        # constants
        n_samples_augmented_min = 150000

        # prepare the X
        # note that since this is an online algorithm, to have better conversion, we augment initial data
        # initiate the Fourier features
        n_samples, n_features = X.shape
        if n_features>D:
            warnings.warn("Input sample is {} dimensional. Fourier space is {} dimensional. NPNN gets better performance when Fourier dimension is higher than the original dimension.".format(n_features, D))
        if n_samples<n_samples_augmented_min:
            # augmentation is necessary
            X_, y_ = self.__augment_data(X=X, y=y, n_samples=n_samples, n_features=n_features, n_samples_augmented_min=n_samples_augmented_min)
            n_samples = X_.shape[0] # get the new data size
        else:
            # augmentation is not necessary
            X_ = X
            y_ = y

        # get number of negative samples after augmentation (if necessary)
        number_of_negative_samples = np.sum(y_==-1)

        # create random fourier feature projections
        alpha = np.random.multivariate_normal(np.zeros((n_features)), 2*g*np.eye(n_features), D).T

        # initiate perceptron parameters
        w = np.random.randn(2*D, 1)*1e-4
        b = np.random.randn(1)*1e-4

        # initiate learning rates
        eta = eta_init
        beta_init = beta_init/number_of_negative_samples # scaling beta init with the number of total negative samples imrpoves convergence
        beta = beta_init
        
        # create negative sample buffer
        # this buffer is used to estimate FPR in an online manner during training, 200 is a constant and can be changed
        negative_sample_buffer_size = max(int(np.round(2/tfpr)), 200)
        negative_sample_buffer = np.zeros((negative_sample_buffer_size))
        negative_sample_buffer_index = 0
        
        # init accumulators
        tp = 0
        fp = 0
        tpr_train_array = np.zeros((n_samples))
        fpr_train_array = np.zeros((n_samples))
        neg_class_weight_train_array = np.zeros((n_samples))
        pos_class_weight_train_array = np.zeros((n_samples))
        gamma_array = np.zeros((n_samples))
        transient_number_of_positive_samples = 1
        transient_number_of_negative_samples = 1

        # initiate weights
        neg_class_weight_train_array[0] = 2*gamma
        pos_class_weight_train_array[0] = 2*gamma

        # SGD
        for i in range(0, n_samples):

            # take the input data
            xt = X_[i,:].reshape(1,-1)
            yt = y_[i]

            # project input to higher dimensional feature space
            xt_r = np.matmul(xt,alpha)
            xt_projected = (1/np.sqrt(D))*np.concatenate([np.cos(xt_r), np.sin(xt_r)], axis=1)

            # make linear prediction
            y_discriminant = np.matmul(xt_projected,w)+b
            yt_predict = np.sign(y_discriminant)[0,0]

            # save tp and fp
            if yt == 1:
                if yt_predict == 1:
                    tp = tp+1
            else:
                if yt_predict == 1:
                    fp = fp+1

            tpr_train_array[i] = tp/transient_number_of_positive_samples
            fpr_train_array[i] = fp/transient_number_of_negative_samples

            # save gamma
            gamma_array[i] = gamma

            # update the buffer with the current prediction
            if yt == -1:

                # modify the size of the FPR estimation buffer
                if negative_sample_buffer_index == negative_sample_buffer_size-1:
                    negative_sample_buffer[:-1] = negative_sample_buffer[1:]
                else:
                    negative_sample_buffer_index += 1

                if yt_predict == 1:
                    # false positive
                    negative_sample_buffer[negative_sample_buffer_index] = 1
                else:
                    # true negative
                    negative_sample_buffer[negative_sample_buffer_index] = 0

            # estimate the FPR of the current model using the moving buffer
            if negative_sample_buffer_index == negative_sample_buffer_size-1:
                estimated_FPR = np.mean(negative_sample_buffer)
            else:
                estimated_FPR = np.mean(negative_sample_buffer[:negative_sample_buffer_index+1])

            # Calculate instant loss
            z = yt*y_discriminant
            dloss_dz = self.__deriv_sigmoid_loss(z, sigmoid_h)
            dz_dw = yt*xt_projected.T
            dz_db = yt
            dloss_dw = dloss_dz*dz_dw
            dloss_db = dloss_dz*dz_db

            # y(t), calculate mu(t)
            if yt == 1:
                # mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                mu = (transient_number_of_positive_samples + transient_number_of_negative_samples)/transient_number_of_positive_samples
                # save class costs
                pos_class_weight_train_array[i] = mu
                if i>1:
                    neg_class_weight_train_array[i] = neg_class_weight_train_array[i-1]
            else:
                # mu(t) uses gamma(t-1), n_plus(t-1), n_minus(t-1)
                mu = gamma*(transient_number_of_positive_samples + transient_number_of_negative_samples)/transient_number_of_negative_samples
                # save class costs
                neg_class_weight_train_array[i] = mu
                if i>1:
                    pos_class_weight_train_array[i] = pos_class_weight_train_array[i-1]

            # update w and b
            w_prev = w
            w = (1-eta*Lambda)*w-eta*mu*dloss_dw
            b = b-eta*mu*dloss_db[0,0] # added [0,0] to have 1D bias

            # update projection
            result_der = (1/np.sqrt(D))*(-w[:D,:].T*np.sin(xt_r) + w[D:,:].T*np.cos(xt_r))
            alpha = alpha - eta*mu*self.__deriv_sigmoid_loss(z,sigmoid_h)*yt*xt.T*result_der

            # update learning rate of perceptron
            eta = eta_init/(1+Lambda*(transient_number_of_positive_samples + transient_number_of_negative_samples))

            # y(t)
            if yt == 1:
                # calculate n_plus(t)
                transient_number_of_positive_samples += 1
            else:
                # calculate n_minus(t)
                transient_number_of_negative_samples += 1
                # calculate gamma(t)
                gamma = gamma*(1+beta*(estimated_FPR - tfpr))

            # update uzawa gain
            beta = beta_init/(1+Lambda*(transient_number_of_positive_samples + transient_number_of_negative_samples))

            # print weight convergence
            if np.mod(i, 100) == 0:
                w_change_norm = np.linalg.norm(w_prev-w)
                w_norm = np.linalg.norm(w)
                diff_perc = abs(w_change_norm)/w_norm*100
                print("{:.4f}%".format(diff_perc))

        # save updated parameters
        self.w_ = w
        self.b_ = b
        self.alpha_ = alpha
            
        # save the results
        self.tpr_train_array_ = tpr_train_array
        self.fpr_train_array_ = fpr_train_array
        self.neg_class_weight_train_array_ = neg_class_weight_train_array
        self.pos_class_weight_train_array_ = pos_class_weight_train_array

        return None

    def predict(self, X):
        # note that input should be numpy array, may need to fix this part for other data types
        n_test = X.shape[0]
        y_predict = np.empty(n_test)
        alpha = self.alpha_
        D = self.D
        w = self.w_
        b = self.b_
        
        for i in range(0, n_test):
            
            # take the input test data
            xt = X[i,:].reshape(1,-1)

            # project input to higher dimensional feature space
            xt_r = np.matmul(xt, alpha)
            xt_projected = (1/np.sqrt(D))*np.concatenate([np.cos(xt_r), np.sin(xt_r)], axis=1)

            # make linear prediction
            y_discriminant = np.matmul(xt_projected,w)+b
            y_predict[i] = np.sign(y_discriminant)[0,0]

        return y_predict

    def score(self, X, y):
        
        y_pred = self.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        FPR = fp/(fp+tn)
        TPR = tp/(tp+fn)
        TFPR = self.tfpr
        np_score = max(FPR, TFPR)/TFPR - TPR
        # since less np_score is better, take negative of the score
        np_score = -np_score

        return np_score

    def get_params(self, deep=True):

        params = dict()
        params['eta_init'] = self.eta_init
        params['beta_init'] = self.beta_init
        params['sigmoid_h'] = self.sigmoid_h
        params['Lambda'] = self.Lambda
        params['D'] = self.D
        params['g'] = self.g
        
        return params

    def set_params(self, **params):

        for key, val in params.items():
            setattr(self, key, val)

        return self

    # aux functions
    def __augment_data(self, X, y, n_samples, n_features, n_samples_augmented_min):
        n_augmentation_call = int(n_samples_augmented_min//n_samples)+1
        n_samples_augmented = int(n_samples*n_augmentation_call)
        X_ = np.empty((n_samples_augmented, n_features))
        y_ = np.empty((n_samples_augmented))
        for i in range(0, n_augmentation_call):
            # shuffle index
            shuffle_index = np.arange(n_samples)
            np.random.shuffle(shuffle_index)
            # create augmented data
            X_[i*n_samples:(i+1)*n_samples, :] = X[shuffle_index, :]
            y_[i*n_samples:(i+1)*n_samples] = y[shuffle_index]
        return X_, y_

    def __deriv_sigmoid_loss(self, z, h):
        sigmoid_loss_x = self.__sigmoid_loss(z, h)
        return h*(1-sigmoid_loss_x)*sigmoid_loss_x

    def __sigmoid_loss(self, z, h):
        return 1/(1+np.exp(-h*z))