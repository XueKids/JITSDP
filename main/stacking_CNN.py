from warnings import filterwarnings
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score
import CNN_models as cnn
from data_preprocess import DataLoader
from sklearn.model_selection import KFold
kfold = KFold(n_splits=2, shuffle=True)

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
dl = DataLoader()
model1 = cnn.SimpleNet()
model2 = cnn.DropNet()

filterwarnings('ignore')

for project in dataset_names:
    x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = dl.build_data(project)
    x_test_scaled = tf.expand_dims(x_test_scaled, -1)

    # Initialize arrays to store predictions
    train_pred = np.zeros((len(x_train), 2))
    test_pred = np.zeros((len(x_test), 2))

    print("---------Project " + project + " begins two-fold cross-validation training-----------")
    count = 0
    for train_idx, val_idx in kfold.split(x_train_scaled):
        count += 1
        # Split data into training and validation sets
        x_train_fold, y_train_fold = x_train_scaled[train_idx], y_train[train_idx]
        x_val_fold, y_val_fold = x_train_scaled[val_idx], y_train[val_idx]
        x_train_fold = tf.expand_dims(x_train_fold, -1)
        x_val_fold = tf.expand_dims(x_val_fold, -1)

        print("---------Begin train base models------------")
        # train base models
        model1.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
        model1.fit(x_train_fold, y_train_fold, batch_size=20, epochs=100, validation_data=(x_val_fold, y_val_fold))
        model1.save('../checkpoint/stackingCNN/stackingbaltwofold_symprod/simpleNet/' + project + '-' + str(count) + '.h5')
        print('\nstep-' + str(count) + ': simpleNet training model is saved')

        model2.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
        model2.fit(x_train_fold, y_train_fold, batch_size=20, epochs=100, validation_data=(x_val_fold, y_val_fold))
        model2.save('../checkpoint/stackingCNN/stackingbaltwofold_symprod/dropNet/' + project + '-' + str(count) + '.h5')
        print('\nstep-' + str(count) + ': dropNet training model is saved')

        # Make predictions on training and test sets using base models
        train_pred[val_idx, 0] = np.squeeze(model1.predict(x_val_fold))
        train_pred[val_idx, 1] = np.squeeze(model2.predict(x_val_fold))
        test_pred[:, 0] += np.squeeze(model1.predict(x_test_scaled))
        test_pred[:, 1] += np.squeeze(model2.predict(x_test_scaled))

    lr = LogisticRegression()
    lr.fit(train_pred, y_train)
    final_preds = lr.predict(test_pred)
    y_MCCscore2 = matthews_corrcoef(y_test, final_preds)
    print(f"\nLR: The MCC_score of the {project} is : {y_MCCscore2}")
    y_F1score = f1_score(y_test, final_preds)
    print(f"LR: The f1_score of the {project} is : {y_F1score}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, final_preds)
    y_auc = metrics.auc(fpr, tpr)
    print(f"LR: The AUC of the {project} is : {y_auc}")