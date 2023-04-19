import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 读取训练集和验证集的 CSV 文件
train= pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\0_train_pre2.csv')
validation = pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\0_validate_pre2.csv')

# 分离特征和标签
X_train = train.iloc[:, :-1]  # 特征
y_train = train.iloc[:, -1]   # 标签

X_val = validation.iloc[:, :-1]  # 特征
y_val = validation.iloc[:, -1]   # 标签

# 尝试不同参数组合
up_fea=10#特征上限
low_fea=1
up_tree=43#树上限
low_tree=1
step_fea=1
step_tree=2
# 记录精度结果
rf_accuracies = np.zeros((int((up_fea-low_fea)/step_fea+1),int((up_tree-low_tree)/step_tree+1)))
#rf_accuracies_b = np.zeros((int((up_tree-low_tree)/step_tree+1),int((up_tree-low_tree)/step_tree+1)))
#svm_accuracies = []
#svm_accuracies_b = []

i=0
for max_features in range(low_fea, up_fea+1,step_fea):
    j=0
    # 划分训练集和测试集
    for n_estimators in range(low_tree,up_tree+1,step_tree):
        # 初始化分类器
        rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=4, min_samples_split=50,random_state=42)
        '''
        n_estimators：随机森林中决策树的数量，默认值为 100。增加这个参数可以提高模型的准确性，但是也会增加训练时间和内存使用量。
        criterion：用于衡量决策树分裂质量的指标。默认值为 "gini"，也可以选择 "entropy"。"gini" 和 "entropy" 都是常用的衡量指标，"gini" 表示基尼不纯度，"entropy" 表示信息熵。
        max_depth：决策树的最大深度。如果不指定，则表示决策树可以无限生长。限制决策树的最大深度可以减少模型过拟合的风险。
        min_samples_split：决策树分裂所需的最小样本数。默认值为 2。当样本数小于该值时，决策树不再进行分裂。
        min_samples_leaf：叶子节点所需的最小样本数。默认值为 1。当叶子节点的样本数小于该值时，不再进行分裂。
        max_features：每个决策树在进行分裂时考虑的最大特征数量。默认值为 "auto"，表示考虑所有特征。可以选择 "sqrt" 或 "log2" 等其他值，表示考虑的特征数量为总特征数量的平方根或以 2 为底的对数。
        bootstrap：是否进行自助采样。默认值为 True。如果设置为 False，则表示不进行自助采样，每个决策树使用全部的样本进行训练。
        '''
            
        # 训练随机森林分类器并计算精度
        rf.fit(X_train, y_train)
        rf_y_pred = rf.predict(X_val)
        #rf_y_pred_b = rf.predict(X_test_b)
        rf_acc = accuracy_score(y_val, rf_y_pred)
        #rf_acc_b = accuracy_score(y_test_b, rf_y_pred_b)
        rf_accuracies[i][j]=rf_acc
        #rf_accuracies_b[i][j]=rf_acc_b
        j+=1
    i+=1
    
# 输出结果
fp = open("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\output_fea.txt", "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
print(f"Random Forest Accuracy:\n{rf_accuracies}",file=fp)
fp.close()

# 将结果保存为图像文件
# 画热图
sns.heatmap(rf_accuracies, cmap='coolwarm',xticklabels=list(range(low_tree, up_tree+1, step_tree)), yticklabels=list(range(low_fea, up_fea+1, step_fea)))

# 添加标题和轴标签
plt.title('Accuracy for test')

plt.xlabel('Number of Features')
plt.ylabel('Number of Estimators')

plt.savefig('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\\Result_fea.png')
