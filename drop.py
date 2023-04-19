import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 从CSV文件读取数据
train_data = pd.read_csv("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\Train.csv")
validate_data = pd.read_csv("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\Validate.csv")
data_list={"train_data":train_data,"validate_data":validate_data}
#用不同的方法补缺，先建立字典

for name ,data in data_list.items():
    #将字符串类型的列转换为数字编码

    labelencoder=LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object' :  # 判断是否为字符串类型
            # 使用Pandas的factorize()方法将字符串编码为数字
            data[col] = labelencoder.fit_transform(data[col])

    # 提取除了最后一列外的全部数据
    data_to_scale = data.iloc[:, :-1]

    # 使用MinMaxScaler对数据进行缩放
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)

    # 将缩放后的数据重新赋值给原始DataFrame中除了最后一列的部分
    data[data.columns[:-1]] = scaled_data

    # 删除训练集中包含缺失值的行
    data.dropna(inplace=True)

#data.to_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\drop_data.csv',na_rep='nan',index=False) # index=False表示不保存索引列，na_rep='nan'使用字符串 'nan' 表示缺失值

# 分离特征和标签
X_train = train_data.iloc[:, :-1]  # 特征
y_train = train_data.iloc[:, -1]   # 标签

X_val = validate_data.iloc[:, :-1]  # 特征
y_val = validate_data.iloc[:, -1]   # 标签

# 尝试不同参数组合
up_fea=min(10,X_train.shape[1])#特征上限
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
