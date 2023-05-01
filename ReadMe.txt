Caution:
The ID of the original data needs to be deleted manually, the program does not handle this
Store the data and change the path in the corresponding code before running the program, no further reminders in the steps
The output will also need to be stored in a new folder.
The following py files, not all have to be run, as required

1, main.py: direct run, through the three methods (mean / median / plural) to fill in the missing data, and a new column of "missing" data, used to record whether the missing, and then RF training, the number of trees and features for a fixed number. Automatically select the solution with the highest accuracy, make predictions, and output the prediction results.
2, analysis: (need the output of step 1), read the data that has been pre-processed by the mean method, output the data feature analysis results, and store
3, RandomForest_features&estimator (change: number of trees, number of features), RandomForest_f&e+PCA (change: number of dimensions, number of trees, number of features), SVM&KNN (that is, SVM & KNN): all based on the output of step 1 (mean value method only), for more detailed attempts and analysis.
4, drop: direct run, after removing the missing data rows, RF training (change: number of trees, number of features), and save accuracy, (not output prediction results, just test accuracy, it is not good so stop)

注意：
原始数据的ID需手动删除，程序对此未做处理
存储数据，并修改相应代码中的路径后，在运行程序，步骤中不再提醒
输出结果也需要设置存储路径/新建文件夹
以下py文件，并不是全部都得运行，按需

1、main.py：直接运行，通过三种方法（均值/中位数/众数）填补缺失数据，并新增一列“missing”的数据，用来记录是否曾缺失，然后RF训练，树数量和特征数为固定。自动选择accuracy最高的补缺方案，进行预测，并输出预测结果。
2、analysis：（需要第1步的输出结果），读取已经通过均值方法预处理过的数据，输出数据特征分析结果，并存储
3、RandomForest_features&estimator（改变：树数量、特征数），RandomForest_f&e+PCA（改变：维数、树数量、特征数），SVM&KNN（就是SVM & KNN）：都是基于步骤1的输出（仅均值方法），进行更加细致的尝试和分析。
4、drop：直接运行，将缺失数据行删除后，RF训练（改变：树数量、特征数），并保存accuracy，（未输出预测结果，只是测试accuracy，不好故作罢）
