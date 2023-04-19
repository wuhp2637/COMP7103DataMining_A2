import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# 读取训练集和验证集的 CSV 文件
train= pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\0_train_pre2.csv')
validation = pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\0_validate_pre2.csv')

# 分离特征和标签
X_train = train.iloc[:, :-1]  # 特征
y_train = train.iloc[:, -1]   # 标签

X_val = validation.iloc[:, :-1]  # 特征
y_val = validation.iloc[:, -1]   # 标签
print("SVM")
#使用PCA进行降维
for component in range(2,X_train.shape[1]):
    pca = PCA(n_components=component) # 设置降维后的维度
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
        
    # 创建 SVM 模型
    svm_model = SVC()

    # 训练 SVM 模型
    svm_model.fit(X_train_pca, y_train)

    # 在验证集上进行预测
    y_pred_val = svm_model.predict(X_val_pca)

    # 计算验证集的准确率
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print(accuracy_val,component)
print("KNN")
#使用PCA进行降维
for component in range(2,X_train.shape[1]):
    pca = PCA(n_components=component) # 设置降维后的维度
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
        
    # 构建KNN模型
    knn = KNeighborsClassifier(n_neighbors=5)  # 创建KNeighborsClassifier对象，设置K值为5

    # 训练模型
    knn.fit(X_train, y_train)

    # 在验证集上进行预测
    y_pred_val = knn.predict(X_val)

    # 计算验证集的准确率
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print(accuracy_val,component)
