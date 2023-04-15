import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import plotly.graph_objects as go
import datapane as dp
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
import time
from collections import defaultdict
# from mlxtend.classifier import StackingCVClassifier
import warnings
# ignore warnings
warnings.filterwarnings("ignore")

## random seed
seed =100
np.random.seed(seed)



train_root = '/home/dengfy/Midterm1/data/training_data.txt'
test_root = '/home/dengfy/Midterm1/data/test_data.txt'


def perturb(data_x):
    # perturb the data by adding random noise
    # data_x: data without label
    # return: perturbed data
    # add random noise
    noise = np.random.normal(0, 0.1, data_x.shape)
    # add noise to data_x
    data_x = data_x + noise
    return data_x


def get_offset(training_data, n=1):
    ## input training_data in original order
    offset_list = []
    offset_list2 = []
    offset_list3 = []
    for i in training_data['subject'].unique():
        subject = training_data[training_data['subject'] == i]
        ## [0:n-1] rows of training_data
        offset = subject[0:subject.shape[0]-1]
        # concate the first row of offset and offset 
        offset = pd.concat([offset[0:1], offset])
        # reset the index of offset same as subject
        idx = subject.index
        offset.index = idx
        # offset column names + .1
        offset.columns = [col + '.1' for col in offset.columns]
        offset_list.append(offset)

        if n>=2:
            offset2 = subject[0:subject.shape[0]-2]
            offset2 = pd.concat([offset2[0:1], offset2[0:1], offset2])
            offset2.index = idx
            offset2.columns = [col + '.2' for col in offset2.columns]
            offset_list2.append(offset2)

        if n>=3:
            offset3 = subject[1:]
            offset3 = pd.concat([offset3, offset3[-1:]])
            offset3.index = idx
            offset3.columns = [col + '.3' for col in offset3.columns]
            offset_list3.append(offset3)

    # concat offset_list
    offset = pd.concat(offset_list)
    if n>=2:
        offset2 = pd.concat(offset_list2)
    if n>=3:
        offset3 = pd.concat(offset_list3)
    # drop subject.1 activity.1
    if 'activity.1' in offset.columns:
        offset = offset.drop(['subject.1', 'activity.1'], axis=1)
    else:
        offset = offset.drop(['subject.1'], axis=1)

    if n>=2:
        if 'activity.2' in offset2.columns:
            offset2 = offset2.drop(['subject.2', 'activity.2'], axis=1)
        else:
            offset2 = offset2.drop(['subject.2'], axis=1)
        if n == 2:
            offset = pd.concat([offset, offset2], axis=1)
    
    if n>=3:
        if 'activity.3' in offset3.columns:
            offset3 = offset3.drop(['subject.3', 'activity.3'], axis=1)
        else:
            offset3 = offset3.drop(['subject.3'], axis=1)
        if n == 3:
            offset = pd.concat([offset, offset2, offset3], axis=1)

    # concate training_data and offset
    s = pd.concat([training_data, offset], axis=1)
    
    return s

def get_data(root, task="task2", train=True, binary=True, offset=False):
    ## task = "task1" or "task2" or "test"


    ## read training_data.txt from second row
    training_data = pd.read_csv(root,sep='\t',skiprows=1, header=None)
    ## open training_data.txt and read the first row
    with open(root) as f:
        first_line = f.readline()
    ## split the first row by tab
    first_line = first_line.split()
    ## change the column name of training_data
    training_data.columns = first_line

    ## number of rows
    print("number of rows:", training_data.shape[0])

    if not train:
        if binary:
            pred = pd.read_csv('/home/dengfy/Midterm1/output/binary_task1.txt', header=None)
            # column name
            pred.columns = ['binary']
            training_data['binary'] = pred['binary']
        if offset:
            training_data = get_offset(training_data, n=offset)
        return training_data

    ## change the activity column
    if task == "task1":
        ## <=3 -> 1   others -> 0
        training_data['activity'] = training_data['activity'].apply(lambda x: 1 if x<=3 else 0)
    elif task == "task2":
        ## >=7 -> 7
        if binary:
            training_data['binary'] = training_data['activity'].apply(lambda x: 1 if x<=3 else 0)
        if offset:
            training_data = get_offset(training_data, n=offset)

        training_data['activity'] = training_data['activity'].apply(lambda x: 7 if x>=7 else x)



    return training_data



def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=seed)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def plot_results(model_scores, name):
    
    model_names = list(model_scores.keys())
    results = [model_scores[model] for model in model_names]
    fig = go.Figure()
    for model, result in zip(model_names, results):
        fig.add_trace(go.Box(
            y=result,
            name=model,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1)
        )
    
    fig.update_layout(
    title='Performance of Different Models Using 10-Fold Cross-Validation',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    xaxis_title='Model',
    yaxis_title='Accuracy',
    showlegend=False)
    # fig.show()
    # save the plot
    fig.write_image(name + ".png")


def model(training_data, task="task2"):
    ## shuffle the order of the data
    training_data = training_data.sample(frac=1, random_state=seed)

    data_y = training_data['activity'] ## y is activity
    data_x = training_data.drop(['activity','subject'], axis=1) ## x is all the other columns

    if False:
        ## oversample
        data_y = data_y.replace(1, 0)
        data_y = data_y.replace(2, 0)
        data_y = data_y.replace(3, 0)
        data_y = data_y.replace(6, 0)
        data_y = data_y.replace(7, 0)
        data_y = data_y.replace(4, 1)
        data_y = data_y.replace(5, 2)

        oversample = ADASYN(sampling_strategy="not majority", random_state=42)
        X_, y_ = oversample.fit_resample(data_x, data_y)
        data_x = X_
        data_y = pd.DataFrame(y_)
        print(data_y.value_counts())
        # index of y = 1 or y=2
        idx = data_y[(data_y['activity']==1) | (data_y['activity']==2)].index
        X_train_smote = data_x.loc[idx]
        Y_train_smote = data_y.loc[idx]
        # 1 -> 3, 2 -> 4
        Y_train_smote = Y_train_smote.replace(1, 4)
        Y_train_smote = Y_train_smote.replace(2, 5)
        print(Y_train_smote.value_counts())

        # combine
        Y_train = training_data['activity'] ## y is activity
        X_train = training_data.drop(['activity','subject'], axis=1) ## x is all the other columns
        idx = Y_train[(Y_train==4) | (Y_train==5)].index
        # drop 3 and 4
        Y_train = Y_train.drop(idx)
        X_train = X_train.drop(idx)
        # add 3 and 4
        # change column name
        Y_train = pd.DataFrame(Y_train)
        Y_train.columns = ['activity']
        Y_train = pd.concat([Y_train, Y_train_smote], axis=0)
        X_train = pd.concat([X_train, X_train_smote], axis=0)
        print(Y_train.value_counts())

        # smote again
        # X_, y_ = oversample.fit_resample(X_train, Y_train)
        # data_x = X_
        # data_y = pd.DataFrame(y_)
        # print(data_y.value_counts())

        data_x = X_train
        data_y = Y_train

    print("######### Stacking ########")
    time_start=time.time()
    if task == "task2":
    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
        # meta_model = GaussianNB()
        meta_model = LogisticRegression(random_state=seed, solver='lbfgs')
        base_models = [
            ('random_forest',RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1,
                           random_state=seed)),
                   ('GaussianNB', GaussianNB()),
                   ('knn1', KNeighborsClassifier(n_neighbors=5)),
                   ('knn2', KNeighborsClassifier(n_neighbors=10)),
                   ('knn3', KNeighborsClassifier(n_neighbors=20)),
                   ('knn4', KNeighborsClassifier(n_neighbors=30)),
                   ('LDA', LinearDiscriminantAnalysis()),
                #    ('QDA', QuadraticDiscriminantAnalysis()),
                   ('gradient_boosting', GradientBoostingClassifier(n_estimators=50, random_state=seed)),
                   ('adaboost', AdaBoostClassifier(n_estimators=50, random_state=seed)),
                    ('svm_rbf', SVC(C=1, kernel='rbf', gamma='auto')),
                    ('svm_linear', SVC(C=1, kernel='linear', gamma='auto')),
                    ('svm_poly', SVC(C=1, kernel='poly', gamma='auto')),
                    ## different small trees: max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, max_features
                    ('decision_tree1', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=5, min_samples_leaf=10, min_samples_split=10, random_state=1)),
                    ('decision_tree2', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=10, min_samples_leaf=20, min_samples_split=20, random_state=2)),
                    ('decision_tree3', DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=5, min_samples_leaf=10, min_samples_split=10, random_state=3)),
                    ('decision_tree4', DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=10, min_samples_leaf=20, min_samples_split=20, random_state=4)),
                    # MLP
                    # ('mlp1', MLPClassifier(hidden_layer_sizes=(2048, 4096, 2048), learning_rate='adaptive', warm_start=True, random_state=seed)),
                    ('mlp2', MLPClassifier(hidden_layer_sizes=(1024, 2048, 1024), learning_rate='adaptive', warm_start=True, random_state=seed)),
                    # ('mlp3', MLPClassifier(hidden_layer_sizes=(512, 1024, 512), learning_rate='adaptive', warm_start=True, random_state=seed)),
                    # ('mlp4', MLPClassifier(hidden_layer_sizes=(256, 512, 256), learning_rate='adaptive', warm_start=True, random_state=seed)),
                ]
        ## stacking
        gs1 = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10, passthrough=True)
        
        gs1.fit(data_x, data_y.squeeze())

    elif task == "task1":
        meta_model = GaussianNB()
        base_models = [
            # ('random_forest',RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1,
            #                random_state=seed)),
            #        ('GaussianNB', GaussianNB()),
                   ('knn', KNeighborsClassifier(n_neighbors=5)),
                   ('LDA', LinearDiscriminantAnalysis()),
                #    ('gradient_boosting', GradientBoostingClassifier(n_estimators=50, random_state=seed)),
                #    ('adaboost', AdaBoostClassifier(n_estimators=50, random_state=seed)),
                # ('svm_rbf', SVC(C=1, kernel='rbf', gamma='auto')),
                    ('svm_linear', SVC(C=1, kernel='linear', gamma='auto')),
                    ('svm_poly', SVC(C=1, kernel='poly', gamma='auto')),
                    # ('tree', DecisionTreeClassifier(max_depth=5, random_state=seed)),
                ]
        ## stacking
        gs1 = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10, passthrough=True)
        gs1.fit(data_x, data_y.squeeze())

    # rf = RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1,
    #                    random_state=seed)
    # model = BaggingClassifier(estimator=rf, n_estimators=10, max_samples=0.9, max_features=1.0, random_state=seed)
    # model.fit(data_x, data_y.squeeze())

    metric = 'accuracy'
    # gs1 = GridSearchCV(estimator = sclf, param_grid = grid, scoring=metric, cv = 10, n_jobs = -1,
    #                 refit=True)
    # gs1.fit(data_x, data_y.squeeze())
    
    # print the best model and the best score
    # print("######### Best model ########")
    # print(gs1.best_params_)
    # print("######### Best score ########")
    # print(gs1.best_score_) # best score
    # confusion matrix
    print("######### Train Confusion matrix ########")
    y_pred = gs1.predict(data_x)
    cm = confusion_matrix(data_y, y_pred)
    print(cm)

    # save the best model and the best score
    # with open("best_model.txt", "a") as f:
    #     f.write(str(gs1.best_estimator_))
    #     f.write("\n")
    #     f.write(metric + ": " + str(gs1.best_score_))
    #     f.write("\n")

    if True:
        model_scores = defaultdict()
        # transform base_models to a dictionary
        # models_dict = {'random_forest': RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1,
        #                    random_state=seed),
        #            'GaussianNB': GaussianNB(),
        #            'knn': KNeighborsClassifier(n_neighbors=5)}
        
        models_dict = dict(base_models)
        for name, model in models_dict.items():
            print('Evaluating {}'.format(name))
            scores = evaluate_model(model, data_x, data_y.squeeze())
            model_scores[name] = scores
        print('Evaluating stacking')
        stacking_scores = evaluate_model(gs1, data_x, data_y.squeeze())
        model_scores['stacking'] = stacking_scores
        plot_results(model_scores, name='stacking_model_cv')
        time_end=time.time()
        time_elapsed = time_end - time_start
        print("Time cost: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        ## write gs1 to best_model.txt
        print("writing to best_model.txt")
        with open("best_model.txt", "a") as f:
            f.write(sid)
            f.write("\n")
            f.write(str(gs1))
            f.write("\n")
            # write model_scores to best_model.txt
            for name, scores in model_scores.items():
                f.write(name + ": " + str(np.mean(scores)))
                f.write("\n")
            f.write("\n")
    return gs1


def get_cm(gs1, training_data):
    ## shuffle the order of the data
    training_data = training_data.sample(frac=1, random_state=seed)

    data_y = training_data['activity'] ## y is activity
    data_x = training_data.drop(['activity','subject'], axis=1) ## x is all the other columns
    ## cross validation confusion matrix
    print("######### Cross validation confusion matrix ########")
    # split the data into 10 folds stratified by the activity
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kf.get_n_splits(data_x, data_y)
    cm_list = []
    # copy the gs1
    tmp = gs1
    ## train on 9 folds and test on 1 fold
    for train_index, test_index in kf.split(data_x, data_y):
        X_train, X_test = data_x.iloc[train_index], data_x.iloc[test_index]
        y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        tmp.fit(X_train, y_train.squeeze())
        y_pred = tmp.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_list.append(cm)
    cm = np.sum(cm_list, axis=0)
    print(cm)
    ## accuracy
    print("######### Cross validation accuracy ########")
    print(np.trace(cm)/np.sum(cm))

    return cm


def predict(model, training_data, test_data, method='stacking'):
    ## shuffle the order of the data
    training_data = training_data.sample(frac=1, random_state=seed)

    data_y = training_data['activity'] ## y is activity
    data_x = training_data.drop(['activity','subject'], axis=1) ## x is all the other columns
    test_x = test_data.drop(['subject'], axis=1) ## x is all the other columns

    if method == 'bagging':
        ## fit 10 models using bagging
        rf = RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1,
                       random_state=42)
        model = BaggingClassifier(estimator=rf, n_estimators=10, max_samples=0.9, max_features=1.0, random_state=seed)
        model.fit(data_x, data_y.squeeze())
    # elif method == 'stacking':
    #     model.fit(data_x, data_y.squeeze())
    
    ## predict the test data
    pred_y = model.predict(test_x)

    return pred_y

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', type=str, required=True)
    args = parser.parse_args()
    sid = args.sid

    task = "task2"
    binary = True
    offset = 2

    training_data = get_data(train_root, task=task, binary=binary, offset=offset)

    clf = model(training_data, task=task)


    test_data = get_data(test_root, task=task, train=False, binary=binary, offset=offset)
    pred_y1 = predict(clf, training_data, test_data, method='bagging')
    pred_y2 = predict(clf, training_data, test_data, method='stacking')
    ## compare pred_y1 and pred_y2
    print("######### Compare bagging and stacking ########")
    print(np.sum(pred_y1 == pred_y2)/len(pred_y1))


    ## save the prediction
    pred_y = pred_y2
    if task == "task1":
        ## output 1: write binary prediction to binary_SID.txt
        with open("output/binary_" + sid + ".txt", "w") as f:
            for i in pred_y:
                f.write(str(i))
                f.write("\n")
        # delete the last \n
        with open("output/binary_" + sid + ".txt", "rb+") as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
    elif task == "task2":
        ## output 2: write multi prediction to multiclass_SID.txt 
        with open("output/multiclass_" + sid + ".txt", "w") as f:
            for i in pred_y:
                f.write(str(i))
                f.write("\n")
        # delete the last \n
        with open("output/multiclass_" + sid + ".txt", "rb+") as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
    
    cm = get_cm(clf, training_data)
    ## output 3: write confusion matrix to confusion_matrix_SID.txt
    with open("output/confusion_matrix_" + sid + ".txt", "w") as f:
        for i in cm:
            for j in i:
                f.write(str(j))
                f.write("\t")
            f.write("\n")
    




