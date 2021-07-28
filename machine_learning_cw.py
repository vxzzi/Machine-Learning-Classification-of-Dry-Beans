from sklearn import preprocessing, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bioinfokit.visuz import cluster
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import randint, loguniform


data = pd.read_excel("Dry_Bean_Dataset.xlsx")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

plt.rcParams["font.family"] = "Times New Roman"

X = data.drop("Class", axis=1)
Y = data['Class']

# Visualizing the correlation between the features
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation between features");plt.show()

####################################################
################ Pre-processing ####################
####################################################
# Detect  outliers in the dataset


def detect_outliers(df, features):
    outlier_indices = []

    for c in features:
        Q1 = np.percentile(df[c], 25)
        Q3 = np.percentile(df[c], 75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)

    return multiple_outliers


data = data.drop(detect_outliers(data,['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity',
                                       'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
                                       'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']), axis=0).reset_index(drop=True)
print('Number of of samples in the dataset after removing outliers: %d' % len(data))

# Bar Chart to visualize the labels in the output variable
var = data['Class']
varValue = var.value_counts()
plt.figure(figsize=(9,3));plt.bar(varValue.index, varValue, color= "blue", edgecolor="yellow", linewidth="2");
plt.xticks(varValue.index, varValue.index.values);plt.ylabel("Frequency");plt.title('Class');
plt.show()

# Convert Class String labels into Integers
lab_enc = preprocessing.LabelEncoder()
label_Y = lab_enc.fit_transform(Y)

# Normalize the input features of the dataset
normalizer = preprocessing.StandardScaler()
norm_X = normalizer.fit_transform(X)


####################################################
############## Feature Extraction ##################
####################################################

# Visualizing the Principal Components in the feature space
pca = PCA()
pca.fit(norm_X)
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = X.columns.values
loadings_df = loadings_df.set_index('variable')

# Screeplot of Principal Components
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_])

# 2D Bi-plot of Principal Components
pca_scores = PCA().fit_transform(norm_X)
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2), var2=round(pca.explained_variance_ratio_[1]*100, 2), colorlist=Y)


# Cumulative Explained Variance Plot
plt.plot(np.cumsum(pca.explained_variance_ratio_)); plt.title('CUMULATIVE EXPLAINED VARIANCE OF THE PRINCIPAL COMPONENTS')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
plt.show()


####################################################
################ MACHINE LEARNING ##################
####################################################

def training_model_metrics(model, X, Y):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=12, shuffle=True)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    model_acc = metrics.accuracy_score(test_y, y_pred)
    f1_measure = metrics.f1_score(test_y, y_pred, average='macro')
    model_precision = metrics.precision_score(test_y, y_pred, average='macro')
    model_recall = metrics.recall_score(test_y, y_pred, average='macro')
    print('Accuracy: %.3f, f1 measure: %.3f, precision: %.3f, recall: %.3f' % (model_acc, f1_measure, model_precision, model_recall))
    metrics.plot_confusion_matrix(model, test_x, test_y);plt.show()

def optimize_param(model, param, X_optim, Y_optim):
    rf_grid = RandomizedSearchCV(estimator=model, n_iter=30, param_distributions=param, scoring='f1_macro', n_jobs=-1,
                                 cv=5, verbose=2, random_state=12)
    print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
    print('-----------------------------------------------------------------------')
    training_model_metrics(rf_grid, X_optim, Y_optim)
    print('The hyper-parameters with the best f1_macro performance:')
    print('----------------------------------------------------------')
    print(rf_grid.best_params_)


def evaluate_PC(model, user_input, user_output):
    train_x, test_x, train_y, test_y = train_test_split(user_input, user_output, test_size=0.2, random_state=12, shuffle=True)
    acc, comp = list(), list()

    for n in range(1, 16):
        pca = PCA(n_components=n)
        pca.fit(train_x)
        pca_transform = pca.fit_transform(train_x)
        cv = KFold(n_splits=5, shuffle=True, random_state=12)
        scores = cross_val_score(model, pca_transform, train_y, scoring='f1_macro', cv=cv, n_jobs=-1)
        acc.append(np.mean(scores))
        comp.append(n)
        print('> No of Components=%d, Accuracy=%.3f' % (n, np.mean(scores)))

    return acc, comp


def display_perf_plot(acc, comp):
    plt.plot(comp, acc)
    plt.title('PRINCIPAL COMPONENT ANALYSIS PERFORMANCE PLOT USING CROSS-VALIDATION')
    plt.axhline(y=max(acc), color='r', linestyle='--')
    plt.xlabel('NUMBER OF COMPONENTS')
    plt.ylabel('F1-MEASURE')
    plt.show()


def KFold_evaluation(model, X, Y):
    means, mins, maxs = list(), list(), list()
    folds = range(2, 13)
    for k in folds:
        cv = KFold(n_splits=k, shuffle=True, random_state=12)
        scores = cross_val_score(model, X, Y, scoring='f1_macro', cv=cv, n_jobs=-1)
        means.append(np.mean(scores))
        mins.append(np.mean(scores) - scores.min())
        maxs.append(scores.max() - np.mean(scores))
    plt.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
    plt.title('CROSS-VALIDATION PERFORMANCE EVALUATION')
    plt.xlabel('NUMBER OF FOLDS')
    plt.ylabel('F1-MEASURE')
    plt.axhline(y=max(means), color='r', linestyle='--')
    plt.show()


####################################################
############ Random Forest Classification ##########
####################################################
print('******************RANDOM FOREST CLASSIFICATION MODEL**************************')
rf_model = RandomForestClassifier(random_state=12)

print('Performance metrics for Random Forest Classification of Original Data')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(rf_model, norm_X, label_Y)


# Using GridSearch to optimize the hyper-parameters in Random Forest Classification Model
n_estimators = randint(100, 2000)
max_features = ['auto', 'sqrt', 'log2']
max_depth = [10, 20, 30, 40, 50]
max_depth.append(None)
min_samples_split = randint(2, 20)
min_samples_leaf = randint(1, 15)

grid_param = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(rf_model, grid_param, norm_X, label_Y)

# PCA Dimensionality Reduction to 8 PCs
print('DIMENSIONALITY REDUCTION USING PRINCIPAL COMPONENT ANALYSIS')
print('-----------------------------------------------------------------------')
print('Performance metrics for Random Forest Classification of Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_rf = PCA(n_components=8).fit_transform(norm_X)
training_model_metrics(rf_model, pca_rf, label_Y)


# Applying SMOTE technique on the dataset
sm = SMOTE(random_state=12)
X_sm, Y_sm = sm.fit_resample(norm_X, label_Y)

# Bar Chart to visualize the labels in the output variable in the SMOTE Balanced Dataset
Y_balanced = lab_enc.inverse_transform(Y_sm)
Y_balanced = pd.Index(Y_balanced, name='Class')
var = Y_balanced
varValue = var.value_counts()
plt.figure(figsize=(9, 3));plt.bar(varValue.index, varValue, color= "blue", edgecolor="yellow", linewidth="2");plt.xticks(varValue.index, varValue.index.values);plt.ylabel("Frequency");plt.title('Class');plt.show()

print('-----------------------------------------------------------------------')
print('Performance metrics for Random Forest Classification of SMOTE Balanced Data')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(rf_model, X_sm, Y_sm)

# Optimize hyper-parameters and display performance metrics for RF model trained using SMOTE data
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(rf_model, grid_param, X_sm, Y_sm)

# PCA Dimensionality Reduction to 8 PCs
print('Performance metrics for Random Forest Classification of SMOTE Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_rfsm = PCA(n_components=8).fit_transform(X_sm)

training_model_metrics(rf_model, pca_rfsm, Y_sm)


####################################################
############ Support Vector Machines ###############
####################################################
print('******************SUPPORT VECTOR MACHINE CLASSIFICATION MODEL**************************')
svm_model = svm.SVC(random_state=12)
print('Performance metrics for Support Vector Machine Classification of Original Data')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(svm_model, norm_X, label_Y)


# Using GridSearch to optimize the hyper-parameters in Support Vector Machine Classification Model
svm_C = [0.1, 1, 10, 100, 1000]
svm_gamma = [1, 0.1, 0.001, 0.00001]
svm_kernel = ['poly', 'rbf', 'linear', 'sigmoid']
svm_degree = randint(1, 10)

svm_param = {'C': svm_C,
             'gamma': svm_gamma,
             'kernel': svm_kernel,
             'degree': svm_degree}

print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(svm_model, svm_param, norm_X, label_Y)


# PCA Dimensionality Reduction to 8 PCs
print('DIMENSIONALITY REDUCTION USING PRINCIPAL COMPONENT ANALYSIS')
print('-----------------------------------------------------------------------')
print('Performance metrics for Support Vector Machine Classification of Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_svm = PCA(n_components=8).fit_transform(norm_X)
training_model_metrics(svm_model, pca_svm, label_Y)

# Applying SMOTE technique on the dataset
print('-----------------------------------------------------------------------')
print('Performance metrics for Support Vector Machine Classification of SMOTE Balanced Data')
print('-----------------------------------------------------------------------')
sm = SMOTE(random_state=12)
X_sm_svm, Y_sm_svm = sm.fit_resample(norm_X, label_Y)
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(svm_model, X_sm_svm, Y_sm_svm)

# Optimize hyper-parameters and display performance metrics for SVM model trained using SMOTE data
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(svm_model, svm_param, X_sm_svm, Y_sm_svm)

# PCA Dimensionality Reduction to 8 PCs
print('Performance metrics for Support Vector Machine Classification of SMOTE Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_sm_svm = PCA(n_components=8).fit_transform(X_sm_svm)
training_model_metrics(svm_model, pca_sm_svm, Y_sm_svm)


####################################################
############ Logistic Regression ###################
####################################################
print('******************LOGISTIC REGRESSION CLASSIFICATION MODEL**************************')
log_reg_model = LogisticRegression(random_state=12, max_iter=400)
print('Performance metrics for Logistic Regression Classification of Original Data')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(log_reg_model, norm_X, label_Y)

# Using GridSearch to optimize the hyper-parameters in Logistic Regression Classification Model
log_reg_C = loguniform(1e-5, 100)
log_reg_penalty = ['none', 'l1', 'l2', 'elasticnet']
log_reg_solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
log_reg_max_iter = [200, 800, 1600, 2000]

log_reg_param = {'C': log_reg_C,
             'penalty': log_reg_penalty,
             'solver': log_reg_solver,
             'max_iter': log_reg_max_iter}

print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(log_reg_model, log_reg_param, norm_X, label_Y)

# PCA Dimensionality Reduction to 8 PCs
print('DIMENSIONALITY REDUCTION USING PRINCIPAL COMPONENT ANALYSIS')
print('-----------------------------------------------------------------------')
print('Performance metrics for Logistic Regression Classification of Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_log_reg = PCA(n_components=8).fit_transform(norm_X)
training_model_metrics(log_reg_model, pca_log_reg, label_Y)

# Applying SMOTE technique on the dataset
print('-----------------------------------------------------------------------')
print('Performance metrics for Logistic Regression Classification of SMOTE Balanced Data')
print('-----------------------------------------------------------------------')
sm = SMOTE(random_state=12)
X_sm_log_reg, Y_sm_log_reg = sm.fit_resample(norm_X, label_Y)
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(log_reg_model, X_sm_log_reg, Y_sm_log_reg)

# Optimize hyper-parameters and display performance metrics for Logistic Regression model trained using SMOTE data
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(log_reg_model, log_reg_param, X_sm_log_reg, Y_sm_log_reg)

# PCA Dimensionality Reduction to 8 PCs
print('Performance metrics for Logistic Regression Classification of SMOTE Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_sm_log_reg = PCA(n_components=8).fit_transform(X_sm_log_reg)
training_model_metrics(log_reg_model, pca_sm_log_reg, Y_sm_log_reg)


####################################################
############ K-Nearest Neighbors###################
####################################################
print('******************K-NEAREST NEIGHBORS CLASSIFICATION MODEL**************************')
knn_model = KNeighborsClassifier()
print('Performance metrics for K-Nearest Neighbors Classification of Original Data')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(knn_model, norm_X, label_Y)

# Using GridSearch to optimize the hyper-parameters in K-Nearest Neighbors Classification Model
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))

knn_param = {'leaf_size': leaf_size,
             'n_neighbors': n_neighbors}
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(knn_model, knn_param, norm_X, label_Y)

# PCA Dimensionality Reduction to 8 PCs
print('DIMENSIONALITY REDUCTION USING PRINCIPAL COMPONENT ANALYSIS')
print('-----------------------------------------------------------------------')
print('Performance metrics for K-Nearest Neighbors Classification of Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_knn = PCA(n_components=8).fit_transform(norm_X)
training_model_metrics(knn_model, pca_knn, label_Y)

# Applying SMOTE technique on the dataset
print('-----------------------------------------------------------------------')
print('Performance metrics for K-Nearest Neighbors Classification of SMOTE Balanced Data')
print('-----------------------------------------------------------------------')
sm = SMOTE(random_state=12)
X_sm_knn, Y_sm_knn = sm.fit_resample(norm_X, label_Y)
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(knn_model, X_sm_knn, Y_sm_knn)

# Optimize hyper-parameters and display performance metrics for K-Nearest Neighbors model trained using SMOTE data
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(knn_model, knn_param, X_sm_knn, Y_sm_knn)

# PCA Dimensionality Reduction to 8 PCs
print('Performance metrics for K-Nearest Neighbors Classification of SMOTE Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_sm_knn = PCA(n_components=8).fit_transform(X_sm_knn)
training_model_metrics(knn_model, pca_sm_knn, Y_sm_knn)


################################################################
############### EXTREME GRADIENT BOOSTING ENSEMBLE #############
################################################################
print('******************XGBOOST CLASSIFICATION MODEL**************************')
xgb_model = XGBClassifier(random_state=12)

print('Performance metrics for XGBoost Classification of Original Data')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(xgb_model, norm_X, label_Y)

# Using GridSearch to optimize the hyper-parameters in XGBoost Classification Model
xgb_param = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
             "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
             "min_child_weight" : [ 1, 3, 5, 7 ],
             "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
             "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(xgb_model, xgb_param, norm_X, label_Y)

# PCA Dimensionality Reduction to 8 PCs
print('DIMENSIONALITY REDUCTION USING PRINCIPAL COMPONENT ANALYSIS')
print('-----------------------------------------------------------------------')
print('Performance metrics for XGBoost Classification of Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_xgb = PCA(n_components=8).fit_transform(norm_X)
training_model_metrics(xgb_model, pca_xgb, label_Y)

# Applying SMOTE technique on the dataset
print('-----------------------------------------------------------------------')
print('Performance metrics for XGBoost Classification of SMOTE Balanced Data')
print('-----------------------------------------------------------------------')
sm = SMOTE(random_state=12)
X_sm_xgb, Y_sm_xgb = sm.fit_resample(norm_X, label_Y)
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using default hyper-parameters')
print('-----------------------------------------------------------------------')
training_model_metrics(xgb_model, X_sm_xgb, Y_sm_xgb)

# Optimize hyper-parameters and display performance metrics for XGBoost model trained using SMOTE data
print('-----------------------------------------------------------------------')
print('Performance Metrics for ML Model of Dataset using optimized hyper-parameters')
print('-----------------------------------------------------------------------')
optimize_param(xgb_model, xgb_param, X_sm_xgb, Y_sm_xgb)

# PCA Dimensionality Reduction to 8 PCs
print('Performance metrics for XGBoost Classification of SMOTE Dataset using 8 Principal Components')
print('-----------------------------------------------------------------------')
pca_sm_xgb = PCA(n_components=8).fit_transform(X_sm_xgb)
training_model_metrics(xgb_model, pca_sm_xgb, Y_sm_xgb)


def disp_KFold():
    model_disp = [rf_model, svm_model, log_reg_model, knn_model, xgb_model]
    for model in model_disp:
        KFold_evaluation(model, norm_X, label_Y)

def disp_PCA():
    acc, comp = evaluate_PC()
    models = [rf_model, svm_model, log_reg_model, knn_model, xgb_model]
    for model in models:
        display_perf_plot(acc, comp)


# Investigate the effect of KFold on the performance of the 5 Models under investigation
print('-----------------------------------------------------------------------')
print('Investigate the effect of KFold on the performance of the Models')
disp_KFold()

# Investigate the effect of the Principal Components on the performance of the 5 Models under investigation
print('-----------------------------------------------------------------------')
print('Investigate the effect of the Principal Components on the performance of the Models')
disp_PCA()