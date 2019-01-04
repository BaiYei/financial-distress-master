import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from sklearn.decomposition import PCA as sklearnPCA

from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm

from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import f1_score


def perf_measure(y_actual, y_hat):  # 性能测量
  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i in range(len(y_hat)):
    if y_actual[i] == y_hat[i] == 1:
       TP += 1
    if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
       FP += 1
    if y_actual[i] == y_hat[i] == 0:
       TN += 1
    if y_hat[i] == 0 and y_actual[i]!=y_hat[i]:
       FN += 1

  print("TP: ", TP, "| FP: ", FP, "| TN: ", TN, " | FN: ", FN)
  print("ERR: ", (float(FP + FN) / float(FP + FN + TP + TN)), "| ACC: ", (float(TP + TN) / float(FP + FN + TP + TN)))
  if (TP + FP):
    print("PREC : ", (float(TP) / float(TP + FP)))
    
  print("Sensitivity (TP Rate)", (float(TP) / float(TP + FN)), "| Specificity (TN Rate)",(float(TN) / float(TN + FP)))
  
  print("F1 Score", f1_score(y_actual, y_hat, average='micro'))
  return (TP, FP, TN, FN)


def knnhelper(x, y):
  xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30, random_state=42,stratify=y)

  neig = knn(n_neighbors=2, algorithm='kd_tree')
  neig = neig.fit(xtrain, ytrain)
  ans = neig.predict(xtest)

  TP,FP,TN,FN =perf_measure(ytest, ans)


def svmhelper(x, y):
  xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

  neig = svm.SVC(gamma=25) # gamma menunjukkan inverse varian
  neig = neig.fit(xtrain, ytrain)
  ans = neig.predict(xtest)

  TP,FP,TN,FN = perf_measure(ytest, ans)


def classfication(x, y, data_description):
    print(data_description, "using KNN")
    knnhelper(x, y)
    print(data_description, "using SVM")
    svmhelper(x, y)


# Start: Read data dari CSV
X_data = np.genfromtxt("Financial Distress.csv", delimiter=",", skip_header=1)[:, 3:]
target = np.genfromtxt("Financial Distress.csv", delimiter=",", skip_header=1,  usecols=2, dtype=int)
# End: Read Data

# #开始:将绝对绝对的绝对特征转化为二进制形式 - Start: Mengubah fitur kategorikal menjadi bentuk binary
# 转置 fitur - transpose fitur
x_trans = np.transpose(X_data)

# 绝对特征类别 - dapetin fitur kategorikal
a = x_trans[79].reshape(-1, 1)

# 删除 - delete fitur kategorikal dari array asli
x_trans = np.delete(x_trans, 79, 0)

# balikin array awal ke bentuk awal  将早期阵列重新连接到原来的形状
X_data = np.transpose(x_trans)

# untuk fitur kategorikal di encode jadi binary  代码的绝对特征是二进制的
enc = OneHotEncoder(categorical_features='all')
enc.fit(a)
a = enc.transform(a).toarray()

# 具有初始特征的类别特征 - concat fitur kategorikal dengan fitur awal
X_data = np.concatenate((X_data, a), axis=1)
# End: 导管功能的改变 - Mengubah fitur kategorikal

# 开始: 数据正常化 - Start: normalisasi data
X_data = normalize(X_data, norm='l2')
# 结束: 数据正常化 End: normalisasi data

classfication(X_data, target, "Data original")

# Start: 使用主成分分析进行降维 - dimension reduction using PCA
# print("PCA前的特征 - Jumlah fitur sebelum PCA",len(X_data[0]))
pca = sklearnPCA(n_components=int(len(X_data[0])*0.80))
X_data = pca.fit_transform(X_data)
# print("PCA后特性的数量 - Jumlah fitur setelah PCA",len(X_data[0]))
# End: 使用主成分分析进行降维

classfication(X_data, target, "Data after PCA")

# # Start: handling imbalance smote-tomek
smtk = SMOTETomek(random_state=0, ratio='auto')
new_data, new_target = smtk.fit_sample(X_data, target)
# # print (new_data.shape)
# # print (np.count_nonzero(new_target==0))
# # print (np.count_nonzero(new_target==1))
# # End: handling imbalance smote-tomek


classfication(new_data, new_target, "Data after Smote-Tomek")

# # 开始: 过采样使用ADASYN - Start: oversampling using adasyn
ada = ADASYN(random_state=12, n_neighbors=3, ratio='auto')
ada_data, ada_target = ada.fit_sample(X_data, target)
# # print('重新取样数据集的形状 - Resampled dataset shape {}'.format(Counter(ada_target)))
# # 结束: 过采样使用ADASYN - End: oversampling using adasyn

classfication(ada_data, ada_target, "使用ADASYN进行过采样后的数据 - Data after oversampling using ADASYN")

# # 开始: 过采样使用SMOTE  -  oversampling using smote
smote = SMOTE(random_state=42, k_neighbors=1, ratio='auto')
sm_data, sm_target = smote.fit_sample(X_data, target)
# # print('重新取样数据集的形状 - Resampled dataset shape {}'.format(Counter(sm_target)))
# # End: 过采样使用SMOTE - oversampling using smote

classfication(sm_data, sm_target, "Data after oversampling using SMOTE")

# # Start: 欠采样使用tomekLink - undersampling using tomekLink
tlink = TomekLinks(random_state=42, ratio='auto')
tl_data, tl_target = tlink.fit_sample(ada_data, ada_target)
print('Resampled dataset shape {}'.format(Counter(tl_target)))
# # # End: 欠采样使用tomekLink - undersampling using tomekLink

classfication(tl_data, tl_target, "ADASYN Data after cleaning using TomekLink")

# # Start: 欠采样使用CondensedNearesNeighbors - undersampling using CondensedNearesNeighbors
enn = EditedNearestNeighbours(random_state=42, n_neighbors=1, ratio='auto')
enn_data, enn_target  = enn.fit_sample(X_data, target)
# # print('重新取样数据集的形状 - Resampled dataset shape {}'.format(Counter(enn_target)))
# # End: 欠采样使用CondensedNearesNeighbors - undersampling using CondensedNearesNeighbors

classfication(enn_data, enn_target, "使用随机采样器进行欠采样后的数据 - Data after under sampling using Edited Nearest Neighbors")

# Start:欠采样使用RandomUnderSampler - undersampling using RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
rus_data, rus_target = rus.fit_sample(X_data, target)
print('Resampled dataset shape {}'.format(Counter(rus_target)))
# End : 欠采样使用RandomUnderSampler - undersampling using RandomUnderSampler

classfication(rus_data, rus_target, "使用随机采样器进行欠采样后的数据 - Data after undersampling using RandomUnderSampler")

# # Start: 不平衡处理使用smote-enn - imbalance handling using smote-enn
smoteenn = SMOTEENN(random_state=42, smote=smote, enn=enn, ratio='auto')
smnn_data, smnn_target = smoteenn.fit_sample(X_data, target)

# # print('重新取样数据集的形状 - Resampled dataset shape {}'.format(Counter(smnn_target)))
# # End: 使用smote-enn处理不平衡 - imbalance handling using smote-enn

classfication(smnn_data, smnn_target, "使用Smote-ENN后的数据 - Data After using Smote-ENN")
