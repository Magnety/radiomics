import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import KFold
from sklearn import svm,neighbors,tree
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

DATA_PATH = 'G:/Breast/Dataset/breast_input/breast_data_153_output_features/'
TRAIN_FEATURES = DATA_PATH + 'features.csv'
TRAIN_LABELS = DATA_PATH + 'labels.csv'
OUTPUT_FEATURES = DATA_PATH + 'output_features.csv'
train_features = pd.read_csv(TRAIN_FEATURES)
train_labels = pd.read_csv(TRAIN_LABELS)
#print(train_labels)

#print(train_labels_noname)
X_cols_name = ['name']
Y_cols = list(train_labels.columns)
#print(X_cols)

xtrain = train_features
#print(xtrain)
ytrain = train_labels
df_train=pd.DataFrame(xtrain)
df_train_name = df_train[X_cols_name]
print(df_train_name)
df= (df_train.drop(columns='name')-df_train.drop(columns='name').min())/(df_train.drop(columns='name').max()-df_train.drop(columns='name').min())
df = pd.DataFrame(df)

df_finall = pd.concat([df_train_name,df],axis=1)
df_finall.to_csv(OUTPUT_FEATURES, index=False, mode='a+')
print(df)



