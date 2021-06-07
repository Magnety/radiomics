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

DATA_PATH = 'G:/tuFramework_data_store/Br_output_features/'
TRAIN_FEATURES = DATA_PATH + 'features.csv'
TRAIN_LABELS = DATA_PATH + 'labels.csv'
OUTPUT_FEATURES = DATA_PATH + 'output_features.csv'
train_features = pd.read_csv(OUTPUT_FEATURES)
df = pd.DataFrame(train_features)
print(df)

