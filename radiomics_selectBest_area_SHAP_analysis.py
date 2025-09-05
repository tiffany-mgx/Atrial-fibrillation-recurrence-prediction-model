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
import shap

warnings.filterwarnings('ignore')


'''
    Radiomics model training
'''


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

# Model training, SHAP
def model(data_x, data_y, common_feature_new_list):
    MODELS = {
        "XGBOOST": xgb.XGBClassifier(n_estimators=500, random_state=618)
    }
    result_model_lst, result_output_lst, result_output_plot_all = list(), list(), list()
    for model_name, model in MODELS.items():
        print(model_name)
        mean_fpr = np.linspace(0, 1, 1000)
        strtfdKFold = StratifiedKFold(n_splits=10)
        kfold = strtfdKFold.split(data_x, data_y)
        tprs, aucs, recalls, fs1, accs, precisions, sensitivitys, specificitys = [], [], [], [], [], [], [], []
        SHAP_values_per_fold = []
        ix_training, ix_test = [], []
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
            # SHAP value
            ix_test.append(test)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_x.iloc[test, :])
            # shap.summary_plot(shap_values, data_x.iloc[test, :], feature_names=common_feature_new_list, max_display=30)

            for SHAPs in shap_values:
                SHAP_values_per_fold.append(SHAPs)
        # Draw SHAP plot
        SHAP_values_arr = np.array(SHAP_values_per_fold)
        new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
        SHAP_data_x_df = data_x.reindex(new_index)
        SHAP_data_x_df.columns = common_feature_new_list

        # shap.plots.beeswarm(np.array(SHAP_values_per_fold), data_x.reindex(new_index))
        shap.summary_plot(SHAP_values_arr,
                          SHAP_data_x_df,
                          feature_names=common_feature_new_list,
                          max_display=50)
                          # max_display=len(common_feature_new_list))
        print(SHAP_values_arr.shape)
        print(SHAP_data_x_df.shape)
        SHAP_values_df = pd.DataFrame(SHAP_values_arr)

        SHAP_values_df.to_csv('result/SHAP_values_df.csv')
        SHAP_data_x_df.to_csv('result/SHAP_data_x_df.csv')

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
    # result_output_df.to_csv('result/result_output_df_radiomics.csv')  # Data ten-fold cross-validation quantitative results
    # result_output_plot_all_df.to_csv('result/result_output_plot_all_df_radiomics.csv')  # Data ten-fold cross-validation quantitative results

    return result_output_df, result_output_plot_all_df


if __name__ == '__main__':
    file_names = ['left_atrium.csv',  # Left atrium
                  'left_aurcle.csv',  # Left atrial appendage
                  'left_ventriculus_snister.csv',  # Left ventricle
                  'right_atrium.csv',  # Right atrium
                  'right_ventriculus_dexter.csv',  # Right ventricle
                  'pulmonary_vein.csv']  # Pulmonary vein

    # >>>>>>>> Use the optimal number of selected features to evaluate the final indicators of the model
    # 1. Common feature selection
    common_feature = pd.read_csv("result/best_common_feature.csv")["common_feature"].tolist()
    print("Common features selected from six areas: ", common_feature)

    # 2. Model training and evaluate by fusion data
    # Read data
    file_name = file_names[0]  # Data file name
    data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
    data_radiomics = data_radiomics[common_feature + ["patient_sn"]]  # Filter common features

    common_feature_new = ['0_' + string for string in common_feature]
    columns_feature_dict = dict(zip(common_feature, common_feature_new))
    data_radiomics.rename(columns=columns_feature_dict, inplace=True)
    data_label = getLabelByDir()
    data_value = pd.merge(data_radiomics, data_label, on='patient_sn', how='left')
    # print("Data dimension after merging, including label:  ", data_value.shape)
    for area_num in range(1, 6):
        # Read data
        file_name = file_names[area_num]  # Data file name
        cur_data_radiomics = pd.read_csv('dataset/' + file_name, encoding="gbk", index_col=0)
        cur_data_radiomics = cur_data_radiomics[common_feature + ["patient_sn"]]  # Filter common features

        common_feature_new = [str(area_num) + '_' + string for string in common_feature]
        columns_feature_dict = dict(zip(common_feature, common_feature_new))
        cur_data_radiomics.rename(columns=columns_feature_dict, inplace=True)

        data_value = pd.merge(data_value, cur_data_radiomics, on='patient_sn', how='left')
        # print("Data dimension after merging, including label:  ", data_value.shape)

    data_value.to_csv('result/data_fusion_six_area.csv')
    # Fusion data standardization
    data_patient_sn = data_value['patient_sn']
    data_y = data_value["label"]
    data_x = data_value.drop(labels=["label", "patient_sn"], axis=1)
    common_feature_new_list = list(data_x.columns)
    z_scaler = preprocessing.StandardScaler()
    data_x = pd.DataFrame(z_scaler.fit_transform(data_x))  # Data standardization
    print("Final data after feature selection: ", data_x.shape)

    # Fusion model training
    result_output_df, result_output_plot_all_df = model(data_x, data_y, common_feature_new_list)




