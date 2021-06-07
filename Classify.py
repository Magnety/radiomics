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
train_features = pd.read_csv(TRAIN_FEATURES)
train_labels = pd.read_csv(TRAIN_LABELS)
#print(train_labels)
train_features_noname = train_features.drop(columns='name')
train_labels_noname = train_labels.drop(columns='name')
#print(train_labels_noname)

X_cols = list(train_features.drop(columns='name',axis=1).columns)
Y_cols = list(train_labels.drop(columns='name',axis=1).columns)
#print(X_cols)
nfolds = 103
kf = KFold(n_splits=nfolds)
pca = PCA(n_components = 50)
mo_base1 = svm.SVC(C=0.1,probability=True,class_weight='balanced')
mo_base2 = neighbors.KNeighborsClassifier(2)
mo_base3 = tree.DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=2000,class_weight='balanced')
mo_base4=  LogisticRegression(penalty="l2")
mo_base5=  GaussianNB()
mo_base6=  LinearDiscriminantAnalysis()
mo_base7 = RandomForestClassifier(n_estimators=2000,class_weight='balanced') #best
mo_base8 = XGBClassifier()
xtrain = train_features_noname[X_cols]
#print(xtrain)
ytrain = train_labels_noname[Y_cols]
df_train=pd.DataFrame(xtrain)
df= (df_train-df_train.min())/(df_train.max()-df_train.min())

#TSNE
"""print(df)
print(np.isnan(df).any())
tsne = TSNE(n_components=3 , init='pca', random_state=501)
df_tsne = tsne.fit_transform(df)
df_tsne = pd.DataFrame(df_tsne)
df_tsne_1= (df_tsne -df_tsne.min())/(df_tsne.max()-df_tsne.min())
plt.figure(figsize=(8, 8))
df_np = np.array(df_tsne_1.loc[:])
y_np = np.array(ytrain.loc[:]).ravel()
print(df_np.shape[0])
for i in range(df_np.shape[0]):
    print("i:",i)
    print(df_np[i,0])
    print(df_np[i,1])
    plt.text(df_np[i, 0], df_np[i, 1], str(y_np[i]), color=plt.cm.Set1(y_np[i]),fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
df = pca.fit_transform(df)"""


df = pd.DataFrame(df)
print(df)

#print(ytrain)
#xtest = test_features[X_cols]
prval = np.zeros(ytrain.shape,dtype=int)
prval.shape
acc=0

y_total = np.zeros((1))
y1_total = np.zeros((1))

for (ff, (id0, id1)) in enumerate(kf.split(df)):
    #print(id0)
    x0, x1 = df.loc[id0], df.loc[id1]
    y0, y1 = np.array(ytrain.loc[id0]), np.array(ytrain.loc[id1])
   # print(".................")
    #print(ytrain.loc[id0])
    softerweight = VotingClassifier(estimators=[('RF',mo_base7),('SVM',mo_base1),('DT',mo_base3)])#

    softerweight.fit(x0, y0.ravel())

    # predicitons
    y = softerweight.predict(x1)
    y_total = np.concatenate((y_total,y))
    y1_total = np.concatenate((y1_total,y1.ravel()))
    print("label:",y1.ravel())
    print("predict:",y)
    accurancy = np.sum(np.equal(y, y1.ravel()))
    print("acc:",accurancy)
    print("////////////////////////////////////")


print("total acc:",accuracy_score(y_total[1:],y1_total[1:]))
print("pre:",precision_score(y_total[1:],y1_total[1:]))
print("rec:",recall_score(y_total[1:],y1_total[1:]))
print("F1:",f1_score(y_total[1:],y1_total[1:]))
print("auc:",roc_auc_score(y_total[1:],y1_total[1:]))