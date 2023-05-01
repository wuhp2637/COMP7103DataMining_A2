import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 从CSV文件读取数据
train_data = pd.read_csv("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\Train.csv")
validate_data = pd.read_csv("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\Validate.csv")
test_data = pd.read_csv("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\Test.csv")

data_list={"train_data":train_data,"validate_data":validate_data,"test_data":test_data}
#用不同的方法补缺，先建立字典
train_data_list=[]
validate_data_list=[]
test_data_list=[]

# 进行预处理
i=0
for name ,data in data_list.items():
# 将字符串类型的列转换为数字编码
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

    # 将空白字符替换为NaN
    data.replace('', np.nan, inplace=True)

    # 创建新列，记录是否曾经缺失
    data.insert(0, 'missing', np.where(data.isnull().any(axis=1), 1, 0))

    # 用平均值填补缺失值
    mean = data.mean(numeric_only=True)
    data1=data.fillna(mean)

    # 用中位数填补缺失值
    median = data.median(numeric_only=True)
    data2=data.fillna(median)

    # 用众数填补缺失值
    mode = data.mode(numeric_only=True).iloc[0]
    data3=data.fillna(mode)

    #用不同的方法补缺，并分别放入不同的集合中
    if name == "train_data":
        train_data_list=[data1, data2, data3]
    if name == "validate_data":
        validate_data_list=[data1, data2, data3]
    if name == "test_data":
        test_data_list=[data1, data2, data3]
    #data.to_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\{}_pre1.csv'.format(name), na_rep='nan', index=False)# index=False表示不保存索引列，na_rep='nan'使用字符串 'nan' 表示缺失值

i=0
for data in train_data_list:
    data.to_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\{}_train_pre2.csv'.format(i),  index=False)# index=False表示不保存索引列，na_rep='nan'使用字符串 'nan' 表示缺失值
    i+=1
i=0
for data in validate_data_list:
    data.to_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\{}_validate_pre2.csv'.format(i),  index=False)# index=False表示不保存索引列，na_rep='nan'使用字符串 'nan' 表示缺失值
    i+=1
i=0
for data in test_data_list:
    data.to_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\{}_test_pre2.csv'.format(i),  index=False)# index=False表示不保存索引列，na_rep='nan'使用字符串 'nan' 表示缺失值
    i+=1

#训练+测试
accuracy=[]
for i in range(3):
    train= train_data_list[i]
    validation = validate_data_list[i]

    # 分离特征和标签
    X_train = train.iloc[:, :-1]  # 特征
    y_train = train.iloc[:, -1]   # 标签

    X_val = validation.iloc[:, :-1]  # 特征
    y_val = validation.iloc[:, -1]   # 标签

    # 创建随机森林分类器并进行训练
    clf = RandomForestClassifier(n_estimators=20,max_features=5,max_depth=4, min_samples_split=50,random_state=42)
    clf.fit(X_train, y_train)

    # 在验证集上进行预测并计算准确度
    y_pred_val = clf.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    accuracy.append(accuracy_val)
max_accuracy = max(accuracy)  # 找到最大准确率值
idx = accuracy.index(max_accuracy)
print (f"准确率：{accuracy}，最优值为第{idx+1}号的{max_accuracy}")
#预测
train= train_data_list[idx]
validation = validate_data_list[idx]

# 分离特征和标签
X_train = train.iloc[:, :-1]  # 特征
y_train = train.iloc[:, -1]   # 标签

X_val = validation.iloc[:, :-1]  # 特征
y_val = validation.iloc[:, -1]   # 标签

# 输出结果
with open('D:/我的文件/learning/sem2/COMP7103 Data mining/assignment 2/output/PredictResult.csv', 'w') as fp:
    fp.write('Predict result:\n')
    for y in y_pred_test:
        if y==0:
            fp.write(f'A\n')
        elif y==1:
            fp.write(f'B\n')
        elif y==2:
            fp.write(f'C\n')
        elif y==3:
            fp.write(f'D\n')

# 读取用于预测的 CSV 文件
test = test_data_list[idx]

# 提取特征并进行预测
X_test = test.iloc[:, :-1]  # 特征
y_pred_test = clf.predict(X_test)  # 预测结果

# 输出结果
fp = open('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 2\output\PredictResult.csv', "a+") # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
print(f"Random Forest Accuracy:\n{y_pred_test}",file=fp)
fp.close()
