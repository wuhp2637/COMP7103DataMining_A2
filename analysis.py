import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# 从CSV文件读取数据
data = pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\0_train_pre2.csv')

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 统计各个分类数量
y_counts = y.value_counts()
print("各个分类的数量：")
print(y_counts)

# 计算各个特征与目标变量的相关性
correlation = data.corr()

# 绘制相关性矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".1%", cmap='coolwarm',annot_kws={"size": 8})
plt.xticks(rotation=-45,fontsize=8) # 设置横坐标标签字号为 8
plt.yticks(rotation=45,fontsize=8) # 设置纵坐标标签字号为 8
plt.title('Correlation Matrix')
plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\correlation.png")
plt.show()

# 主成分分析
pca = PCA()
pca.fit_transform(X)
covariance=pca.get_covariance()
explained_variance=pca.explained_variance_
ratio=pca.explained_variance_ratio_

# 可视化直方图
plt.figure(figsize=(6,4),dpi=150) 
plt.bar(range(data.shape[1]-1), explained_variance, alpha=0.5, align='center')
plt.ylabel('Explained variance')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\PCA Explained variance ratio.png")

# 可视化 解释方差比例之和
plt.clf()
plt.plot([i for i in range(X.shape[1])],[sum(ratio[:i+1]) for i in range(X.shape[1])])
plt.ylabel('Accumulated explained variance ratio')
plt.xlabel('Principal components')
plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\PCA Accumulated explained variance ratio.png")
'''
# 选择需要绘制直方图的变量
for i in range(data.shape[1]-1):
    # 绘制直方图
    plt.clf()
    plt.hist(data.iloc[:, i].values)
    plt.title('Distribution of '+ data.columns[i])
    plt.xlabel('values')
    plt.ylabel('Frequency')
    plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\" + str(i) + "_hist.png")
    plt.show()
    '''