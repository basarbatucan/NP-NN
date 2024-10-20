import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from npnn import npnn

# Target False Alarm
# NP framework aims to maximize the detection power while upper bounding the false alarm rate
# target false alarm rate should be determined by the user
target_FPR = 0.9

# main 
# np-nn works for 1,-1 classification
# we expect data to be in tabular form with the latest column as target (check ./data/banana.csv)
data = pd.read_csv('./data/tmp.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# define hyperparameters
parameters = {
    'eta_init': [0.01],               # default, 0.01
    'beta_init': [100],               # default, 100
    'sigmoid_h': [-1],            # default, -1
    'Lambda':[0],               # default, 0
    'D':[2],                       # default, 2
    'g':[0.1]                      # default, 1
    }

# define classifier
NPNN = npnn(tfpr=target_FPR)

# hyperparameter tuning
clf = GridSearchCV(NPNN, parameters, verbose=3, cv=2, n_jobs=-1)

# training
clf.fit(X_train, y_train)

# print best params
print(clf.best_params_)

# get best estimator
best_NPNN = clf.best_estimator_

# prediction
y_pred = best_NPNN.predict(X_test)

# evaluation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
FPR = fp/(fp+tn)
TPR = tp/(tp+fn)
print("NPNN, TPR: {:.3f}, FPR: {:.3f}".format(TPR, FPR))

# plot transient performances
f,ax = plt.subplots(3,1,figsize=(8,12))
ax[0].plot(best_NPNN.tpr_train_array_, label="TPR")
ax[0].set_xlabel("Number of Samples")
ax[0].set_ylabel("TPR")
ax[0].grid()
ax[0].legend()
ax[1].plot(best_NPNN.fpr_train_array_, label="FPR")
ax[1].set_xlabel("Number of Samples")
ax[1].set_ylabel("FPR")
ax[1].grid()
ax[1].legend()
ax[2].plot(best_NPNN.neg_class_weight_train_array_, label="Negative Class weight")
ax[2].plot(best_NPNN.pos_class_weight_train_array_, label="Positive Class weight")
ax[2].set_xlabel("Number of Samples")
ax[2].set_ylabel("Class Respective Weights")
ax[2].grid()
ax[2].legend()
f.savefig('./figures/transient_performances.png')

# plot decision boundaries if the input is 2 dimensional
# create a mesh to plot in
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
X_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = best_NPNN.predict(X_mesh)
Z = Z.reshape(xx.shape)
pos_class_indices = (y_test == 1)
neg_class_indices = (y_test == -1)
fp_indices = (y_test==-1) & (y_pred==1)
f,ax = plt.subplots(1,1,figsize=(12,8))
ax.scatter(X_test[pos_class_indices,0], X_test[pos_class_indices,1], marker='x', c='b', label="Class 1")
ax.scatter(X_test[neg_class_indices,0], X_test[neg_class_indices,1], marker='o', c='r', label = "Class -1")
ax.scatter(X_test[fp_indices,0], X_test[fp_indices,1], marker='*', c='g', label="False alarm")
ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
ax.legend()
ax.grid()
ax.set_ylabel("X_2")
ax.set_xlabel("X_1")
ax.set_title("TFPR:{:.3f}".format(best_NPNN.tfpr))
f.savefig('./figures/decision_boundary_visualized.png')