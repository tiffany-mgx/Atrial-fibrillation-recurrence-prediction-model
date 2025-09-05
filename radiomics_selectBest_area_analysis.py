# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, roc_auc_score
# from torchmetrics import Specificity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import preprocessing
from xgboost import plot_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import torch
from itertools import cycle
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve_myself

warnings.filterwarnings('ignore')


'''
    Model training and evaluation
'''

area_num = 0

# Get labels for case and control groups
def getLabelByDir():
    case_path = 'case path'
    control_path = 'control path'
    case_names = os.listdir(case_path)
    control_names = os.listdir(control_path)
    label = [1 for _ in range(len(case_names))] + [0 for _ in range(len(control_names))]
    patient_sn = case_names + control_names
    dic = {'label': label,
           'patient_sn': patient_sn
           }
    data_label = pd.DataFrame(dic)
    return data_label


# Feature selection using filtering method: correlation coefficient method
def selectImportFeature(data_x, data_y, feature_size):
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=feature_size)
    matrix = selector.fit_transform(data_x, data_y)
    featurelst = selector.get_feature_names_out()
    # print(selector.pvalues_)
    # print(selector.scores_)
    return featurelst


# Model training, manually split the dataset and fit, the advantage is that it retains the parameters of each training model
def model(data_x, data_y):
    MODELS = {
        "XGBOOST": xgb.XGBClassifier(n_estimators=500, random_state=618),
        "RF": RandomForestClassifier(n_estimators=500, random_state=618),
        'Bayesian': GaussianNB(),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'SVM': SVC(random_state=618),
        'Logistic': LogisticRegression(random_state=618)
    }

    result_model_lst, result_output_lst, result_output_plot_all = list(), list(), list()
    for model_name, model in MODELS.items():
        print(model_name)
        mean_fpr = np.linspace(0, 1, 1000)
        # Create an instance of StratifiedKFold to get different training and testing sample indices, with 10 folds
        strtfdKFold = StratifiedKFold(n_splits=10)
        kfold = strtfdKFold.split(data_x, data_y)
        tprs, aucs, recalls, fs1, accs, precisions, sensitivitys, specificitys = [], [], [], [], [], [], [], []
        for i, (train, test) in enumerate(kfold):
            model.fit(data_x.iloc[train, :], data_y.iloc[train])
            fpr, tpr, roc_auc, recall, f1, acc, precision, sensitivity, specificity = plot_roc_curve_myself(model, data_x.iloc[test, :], data_y.iloc[test])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            fs1.append(f1)
            accs.append(acc)
            sensitivitys.append(sensitivity)
            specificitys.append(specificity)

        result_model_lst.append([model_name, tprs, mean_fpr, aucs])
        result_output_plot_all.append([model_name, accs, sensitivitys, specificitys, fs1, aucs])   # Output all cross-validation results and plot box plots
        result_output_lst.append([
            model_name,
            np.median(accs), str(round(np.percentile(accs, 2.5), 2)) + "-" + str(round(np.percentile(accs, 97.5), 2)),
            np.median(sensitivitys),
                             str(round(np.percentile(sensitivitys, 2.5), 2)) + "-" + str(round(np.percentile(sensitivitys, 97.5), 2)),
            np.median(specificitys),
                             str(round(np.percentile(specificitys, 2.5), 2)) + "-" + str(round(np.percentile(specificitys, 97.5), 2)),
            np.median(fs1), str(round(np.percentile(fs1, 2.5), 2)) + "-" + str(round(np.percentile(fs1, 97.5), 2)),
            np.median(aucs), str(round(np.percentile(aucs, 2.5), 2)) + "-" + str(round(np.percentile(aucs, 97.5), 2))])

        print("Average Cross Validation score ACC:{}, sensitivity:{}, specificity:{}, f1:{} , roc_auc:{}".format(
            np.median(accs), np.median(sensitivitys), np.median(specificitys), np.median(fs1), np.median(aucs)))

    result_output_df = pd.DataFrame(result_output_lst,
                                    columns=["model_name",
                                             "ACC", "ACC_CI",
                                             "sensitivity", "sensitivity_CI",
                                             "specificity", "specificity_CI",
                                             "F1", "F1_CI",
                                             "AUC", "AUC_CI"])
    result_output_plot_all_df = pd.DataFrame(result_output_plot_all,
                                             columns=["model_name", "ACC", "Sensitivity", "Specificity", "F1", "AUC"])
    # result_output_df.to_csv('result/result_output_df_radiomics.csv')
    # result_output_plot_all_df.to_csv('result/result_output_plot_all_df_radiomics.csv') 

    return result_output_df, result_output_plot_all_df, result_model_lst


# Draw ROC curve
def plot_roc_confidence(result_model_lst):
    plt.figure("Result confidence")
    plt.rcParams['axes.facecolor'] = 'lightblue'
    i, colors = 0, ['red', 'chocolate', 'olive', 'lightseagreen', 'steelblue', 'blueviolet']   # color=colors[i],
    for model_name, tprs, mean_fpr, aucs in result_model_lst:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        tprs_upper = np.percentile(tprs, 97.5)
        tprs_lower = np.percentile(tprs, 2.5)
        plt.plot(mean_fpr, mean_tpr, lw=1, linestyle='-', alpha=1.0, color=colors[i],
                 label='{0} : AUC={1:0.2f}(95% CI:{2:0.2f} - {3:0.2f})'.
                 format(model_name, np.median(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
                 )

        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='black')
        i = i+1

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    font = {'family': 'Times New Roman',  # 'weight': 'bold',
            'size': 18}
    plt.rc('font', **font)
    plt.xlabel('Specificity', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Sensitivity', fontdict={'family': 'Times New Roman', 'size': 18})
    # ax.set_facecolor('#eafff5')
    plt.tick_params(labelsize=16)
    plt.legend(loc="lower right", fontsize=10, frameon=False)
    plt.show()


def optimalFeatureSelection(file_names):
    best_auc = 0
    model_xgb_roc_list, model_rf_roc_list, model_bays_roc_list, model_knn_roc_list, model_svm_roc_list, model_log_roc_list = list(), list(), list(), list(), list(), list()
    for feature_size in range(50, 112):
        print("\nCurrent filtered feature count: ", feature_size+1)
        # 1. Common feature selection
        feature_area_list = list()
        for area_num in range(6):
            # Read data
            file_name = file_names[area_num]  # Data file name
            data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
            data_label = getLabelByDir()
            data_value = pd.merge(data_radiomics, data_label, on='patient_sn', how='left')
            # Data standardization
            data_patient_sn = data_value['patient_sn']
            data_y = data_value["label"]
            data_x = data_value.drop(labels=["label", "patient_sn"], axis=1)
            z_scaler = preprocessing.StandardScaler()
            data_x = pd.DataFrame(z_scaler.fit_transform(data_x), columns=data_x.columns.tolist())  # Data standardization
    
            # Feature selection
            featurelst = selectImportFeature(data_x, data_y, feature_size=feature_size)
            feature_area_list.append(list(featurelst))
        # print("Features selected from six areas: ", feature_area_list)
        common_feature = list(set(feature_area_list[0]) & set(feature_area_list[1]) & set(feature_area_list[2]) &
                               set(feature_area_list[3]) & set(feature_area_list[4]) & set(feature_area_list[5]))
        print("Number of common features selected from six areas: ", len(common_feature))
    
        # 2. Model training and evaluate by fusion data
        # Read data
        file_name = file_names[0]  # Data file name
        data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
        data_radiomics = data_radiomics[common_feature + ["patient_sn"]]  # Filter common features
        data_label = getLabelByDir()
        data_value = pd.merge(data_radiomics, data_label, on='patient_sn', how='left')
        # print("Data dimension after merging, including label:  ", data_value.shape)
        for area_num in range(1, 6):
            # Read data
            file_name = file_names[area_num]  # Data file name
            cur_data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
            cur_data_radiomics = cur_data_radiomics[common_feature + ["patient_sn"]]  # Filter common features
    
            data_value = pd.merge(data_value, cur_data_radiomics, on='patient_sn', how='left')

        # Fusion data standardization
        data_patient_sn = data_value['patient_sn']
        data_y = data_value["label"]
        data_x = data_value.drop(labels=["label", "patient_sn"], axis=1)
        z_scaler = preprocessing.StandardScaler()
        data_x = pd.DataFrame(z_scaler.fit_transform(data_x))  # Data standardization
        print("Final data after feature selection: ", data_x.shape)
    
        # Fusion model training
        result_output_df, result_output_plot_all_df = model(data_x, data_y)
        if result_output_df['AUC'].iloc[0] > best_auc:  # Based on the XGBOOST model, get the best feature selection count result
            best_auc = result_output_df['AUC'].iloc[0]
            result_output_df.to_csv('result/result_output_df_radiomics.csv')  # Data ten-fold cross-validation quantitative results
            result_output_plot_all_df.to_csv('result/result_output_plot_all_df_radiomics.csv')  # Data ten-fold cross-validation quantitative results
            pd.DataFrame({"common_feature": common_feature}).to_csv("result/best_common_feature.csv")
    
        # Save evaluate result (feature size and roc value), and plot curve
        model_xgb, model_rf, model_bays, model_knn, model_svm, model_log = result_output_df['AUC'].iloc[0], result_output_df['AUC'].iloc[1], \
                                                                           result_output_df['AUC'].iloc[2], result_output_df['AUC'].iloc[3], \
                                                                           result_output_df['AUC'].iloc[4], result_output_df['AUC'].iloc[5]
        model_xgb_roc_list.append(model_xgb)
        model_rf_roc_list.append(model_rf)
        model_bays_roc_list.append(model_bays)
        model_knn_roc_list.append(model_knn)
        model_svm_roc_list.append(model_svm)
        model_log_roc_list.append(model_log)
    
    feature_size_and_roc_df = pd.DataFrame({"xgboost": model_xgb_roc_list, "rf": model_rf_roc_list, "bayesian": model_bays_roc_list,
                                            "knn": model_knn_roc_list, "svm": model_svm_roc_list, "logistic": model_log_roc_list})
    feature_size_and_roc_df.to_csv('result/result_feature_size_and_roc_df.csv')
    # Draw feature size and roc value chart
    plt.figure(0)
    plt.plot(model_xgb_roc_list, label='xgboost', marker='o')
    plt.plot(model_rf_roc_list, label='rf', marker='o')
    plt.plot(model_bays_roc_list, label='bayesian', marker='o')
    plt.plot(model_knn_roc_list, label='knn', marker='o')
    plt.plot(model_svm_roc_list, label='svm', marker='o')
    plt.plot(model_log_roc_list, label='logistic', marker='o')
    plt.legend()
    plt.show()

# Use the optimal number of selected features to evaluate the final indicators of the model
def fusionModel(file_names):
    # 1. Common feature selection
    common_feature = pd.read_csv("result/best_common_feature.csv")["common_feature"].tolist()
    print("Common features selected from six areas: ", common_feature)

    # 2. Model training and evaluate by fusion data
    # Read data
    file_name = file_names[0]  # Data file name
    data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
    data_radiomics = data_radiomics[common_feature + ["patient_sn"]]  # Filter common features
    data_label = getLabelByDir()
    data_value = pd.merge(data_radiomics, data_label, on='patient_sn', how='left')
    # print("Data dimension after merging, including label:  ", data_value.shape)
    for area_num in range(1, 6):
        # Read data
        file_name = file_names[area_num]  # Data file name
        cur_data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
        cur_data_radiomics = cur_data_radiomics[common_feature + ["patient_sn"]]  # Filter common features
        data_value = pd.merge(data_value, cur_data_radiomics, on='patient_sn', how='left')

    # print(data_value.columns)
    # Fusion data standardization
    data_patient_sn = data_value['patient_sn']
    data_y = data_value["label"]
    data_x = data_value.drop(labels=["label", "patient_sn"], axis=1)
    z_scaler = preprocessing.StandardScaler()
    data_x = pd.DataFrame(z_scaler.fit_transform(data_x))  # Data standardization
    print("Final data after feature selection: ", data_x.shape)

    # Fusion model training
    result_output_df, result_output_plot_all_df, result_model_lst = model(data_x, data_y)
    print(result_model_lst)
    # All models' ROC curve area and confidence interval range
    plot_roc_confidence(result_model_lst)
    return common_feature

# Use the optimal number of selected features to evaluate the performance indicators of each cardiac region model
def compareModel(file_names):
    # 3. Model training and evaluate by separately data
    for area_num in range(6):
        print("Cardiac region number: ", area_num)
        # Read data
        file_name = file_names[area_num]  # Data file name
        data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)

        data_label = getLabelByDir()
        data_value = pd.merge(data_radiomics, data_label, on='patient_sn', how='left')

        # Data standardization
        data_patient_sn = data_value['patient_sn']
        data_y = data_value["label"]
        data_x = data_value.drop(labels=["label", "patient_sn"], axis=1)
        data_x = data_x[common_feature]
        z_scaler = preprocessing.StandardScaler()
        data_x = pd.DataFrame(z_scaler.fit_transform(data_x), columns=common_feature)  # Data standardization
        print("Final data after feature selection: ", data_x.shape)

        # Model training
        result_output_df, result_output_plot_all_df, result_model_lst = model(data_x, data_y)
        result_output_df.to_csv('result/result_output_df_radiomics_' + str(area_num) + '.csv')  # Data ten-fold cross-validation quantitative results

if __name__ == '__main__':
    file_names = ['left_atrium.csv',  # Left atrium
                  'left_aurcle.csv',  # Left atrial appendage
                  'left_ventriculus_snister.csv',  # Left ventricle
                  'right_atrium.csv',  # Right atrium
                  'right_ventriculus_dexter.csv',  # Right ventricle
                  'pulmonary_vein.csv']  # Pulmonary vein
    
    # optimalFeatureSelection(file_names)

    common_feature = fusionModel(file_names)

    compareModel(file_names, common_feature)
    
