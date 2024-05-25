import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from data_preprocess import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,GradientBoostingClassifier

warnings.filterwarnings("ignore")

dataset_names = [
                  'afwall',
                  'alfresco',
                  'AnySoftKeyboard',
                  'apg',
                  'Applozic',
                  'apptentive',
                  'atmosphere',
                  'chat_secure',
                  'deltachat',
                  'facebook',
                  'image',
                  'kiwix_android',
                  'lottie',
                  'openxc',
                  'own_cloud',
                  'PageTurner',
                  'signal',
                  'sync',
                  'syncthing',
                  'wallpaper',
                 ]

columns = ['fix', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt',
           'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'contains_bug']
dl = DataLoader()

for project in dataset_names:
    X_for_generate_1, X_for_generate_0, x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = dl.build_data(project)

    svc_model = SVC()
    svc_model.fit(x_train_scaled, y_train)
    svc_pred = svc_model.predict(x_test_scaled)
    svm_MCC = matthews_corrcoef(svc_pred, y_test)
    print(f"SVM: The MCC_score of the {project} is : {svm_MCC}")
    y_F1score = f1_score(y_test, svc_pred)
    print(f"SVM: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, svc_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"SVM: The AUC of the {project} is: {y_auc}")

    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(x_train_scaled, y_train)
    naive_bayes_pred = naive_bayes_model.predict(x_test_scaled)
    nb_mcc = matthews_corrcoef(naive_bayes_pred, y_test)
    print(f"NB: The MCC_score of the {project} is : {nb_mcc}")
    y_F1score = f1_score(y_test, naive_bayes_pred)
    print(f"NB: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, naive_bayes_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"NB: The AUC of the {project} is: {y_auc}")

    logistic_model = LogisticRegression()
    logistic_model.fit(x_train_scaled, y_train)
    lr_pred = logistic_model.predict(x_test_scaled)
    lr_mcc = matthews_corrcoef(lr_pred,y_test)
    print(f"LR: The MCC_score of the {project} is : {lr_mcc}")
    y_F1score = f1_score(y_test, lr_pred)
    print(f"LR: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, lr_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"LR: The AUC of the {project} is: {y_auc}")

    tree_model = DecisionTreeClassifier()
    tree_model.fit(x_train_scaled, y_train)
    dt_pred = tree_model.predict(x_test_scaled)
    dt_mcc = matthews_corrcoef(dt_pred, y_test)
    print(f"DT: The MCC_score of the {project} is : {dt_mcc}")
    y_F1score = f1_score(y_test, dt_pred)
    print(f"DT: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, dt_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"DT: The AUC of the {project} is: {y_auc}")

    KNN = KNeighborsClassifier()
    KNN.fit(x_train_scaled, y_train)
    knn_pred = KNN.predict(x_test_scaled)
    knn_mcc = matthews_corrcoef(knn_pred, y_test)
    print(f"KNN: The MCC_score of the {project} is : {knn_mcc}")
    y_F1score = f1_score(y_test, knn_pred)
    print(f"KNN: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, knn_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"KNN: The AUC of the {project} is: {y_auc}")

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train_scaled, y_train)
    RF_pred = clf.predict(x_test_scaled)
    random_forest_mcc = matthews_corrcoef(RF_pred, y_test)
    print(f"RF: The MCC_score of the {project} is : {random_forest_mcc}")
    y_F1score = f1_score(y_test, RF_pred)
    print(f"RF: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, RF_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"RF: The AUC of the {project} is: {y_auc}")

    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train_scaled, y_train)
    GB_pred = clf1.predict(x_test_scaled)
    gb_forest_mcc = matthews_corrcoef(GB_pred, y_test)
    print(f"GB: The MCC_score of the {project} is : {gb_forest_mcc}")
    y_F1score1 = f1_score(y_test, GB_pred)
    print(f"GB: The f1_score of the {project} is : {y_F1score1}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, GB_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"GB: The AUC of the {project} is: {y_auc}")

    clf2 = AdaBoostClassifier(n_estimators=100, random_state=0).fit(x_train_scaled, y_train)
    AdB_pred = clf2.predict(x_test_scaled)
    ab_forest_mcc = matthews_corrcoef(AdB_pred, y_test)
    print(f"AdB: The MCC_score of the {project} is : {ab_forest_mcc}")
    y_F1score = f1_score(y_test, AdB_pred)
    print(f"AdB: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, AdB_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"AdB: The AUC of the {project} is: {y_auc}")

    clf3 = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(x_train_scaled, y_train)
    bagging_pred = clf3.predict(x_test_scaled)
    bagging_mcc = matthews_corrcoef(bagging_pred, y_test) 
    print(f"Bagging: The MCC_score of the {project} is : {bagging_mcc}")
    y_F1score = f1_score(y_test, bagging_pred)
    print(f"Bagging: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, bagging_pred)
    y_auc = metrics.auc(fpr, tpr)
    print(f"Bagging: The AUC of the {project} is: {y_auc}")