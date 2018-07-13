# Ian Lyons
# Assignment 2
# ML PIPELINE
'''

'''
import inspect
import os
import re
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.tree as tree

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

def camel_to_snake(camelcase):
    """ 
    This function is taken from https://gist.github.com/jaytaylor/3660565. 
    """
    _underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
    _underscorer2 = re.compile('([a-z0-9])([A-Z])')
    subbed = _underscorer1.sub(r'\1_\2', camelcase)
    return _underscorer2.sub(r'\1_\2', subbed).lower()

def read_csv(path):
    '''
    '''
    if os.path.exists:
        df = pd.read_csv(path)
        return df
    else:     
        raise Exception('Check that path exists')


def fill_na(df, na_cols, value=None):
    if value is not None:
        df[col].fillna(value, inplace=True)
        return df
    else:
        for col in na_cols:
            col_median = df[col].median()
            df[col].fillna(col_median, inplace = True)
            col_median = None
        return df


def plot_distribution(df, targetcol, xlabel, ylabel, title, num_bins):
    plt.hist(df[targetcol], bins=num_bins)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    return plt.show()


def sns_histogram(df, targetcol, title):
    targetvar = df[targetcol]
    sns.distplot(targetvar)
    plt.title(title)
    plt.ylabel('Frequency')
    return plt.show()


def plot_scatter(df, x_ax, y_ax, title):
    '''
    '''
    plt.scatter(df[x_ax], df[y_ax])
    plt.title(title)
    plt.xlabel(x_ax) 
    plt.ylabel(y_ax)
    return plt.show()

def sns_countchart(df, targetcol):
    '''
    '''
    targetvar = df[targetcol]
    sns.countplot(targetvar, data=df)
    plt.title(targetcol + ' Countplot')
    return plt.show()

def outliers(df, col, zparam=1.96):
    '''    
        col is the column name
        zparam (float) the z-score to consider an outlier. Defaults to 1.96 (p=0.05)
        
    RETURNS:
        df (pd dataframe) that has been updated
        newcol (str) name of the new column
        outliers (list) list of indices pertaining to outlier set
    '''
    newcol = str(col) + "_zscore"
    df[newcol] = df[col].apply(lambda x: (x - df[col].mean()) / df[col].std())
    outliers = df.index[abs(df[newcol]) >= zparam ].tolist()
    return df, newcol, outliers

def discretize_with_quartiles(df, var, new_var, num_bins, labels=None):
    '''
    Discretize a continuous variable, var. Num_bins is the number of bins to break up the target 
    column into. If labels are provided, then the new bins will have the label names. 
    '''
    if labels:
        df[new_var] = pd.qcut(x=df[var], q=num_bins, labels=labels)
    else:
        df10[new_var] = pd.qcut(x=df[var], q=num_bins)
    return df

def make_discretized(df, var_of_interest, num_buckets=4):
    '''
    Discretizes a continuous variable
    
    inputs: 
     var_of_interest: str
     num_buckets 
    
    returns: 
    df (pd dataframe) with appended categorical variable columns
    '''
    col_of_interest = df[var_of_interest]
    new_col_name = str(col_of_interest) + '_categorized'
    df[new_col_name] = pd.cut(col_of_interest, num_buckets)
    df_append = pd.get_dummies(data=df[new_col_name])
    df = pd.concat([df, df_append], axis=1)
    return df

def create_dummy(df, var, dummy, lambda_):
    '''
    Creates a dummy variable from column var using the condition in lambda_. 
    '''
    df[dummy] = df[var].apply(lambda_)
    return df


def split_train_test(df, outcome_var, feature_list, test_fraction, random_state=42):
    '''
    Splits the dataset into a training set and a test set. 
    - test_fraction: a float between 0 and 1 that specifies the fraction of 
    the dataset to set aside for testing. 
    '''
    X = df[feature_list]
    Y = df[outcome_var]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    return x_train, x_test, y_train, y_test


def evaluate_classifier(classifier, x_train, y_train, x_test, y_test):
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    print(' Training Score: {} \n Testing Score: {}'.format(train_score, test_score))


def test_different_k(classifier, x_train, y_train, x_test, y_test, k_range):
    '''
    Iterates through values for k, storing the accuracy scores. Creates a plot of the k values against
    their accuracy score. 
    Returns the value of k with the highest accuracy score. 
    To do: Implement more measures of validity than accuracy. 
    '''
    lower = k_range[0]
    upper = k_range[1]
    plot_data_train = dict()
    for k in range(lower, upper):
        if k <= 0:
            k = 1
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', metric_params={'p': 3})
        knn.fit(x_train, y_train)
        train_pred = knn.predict(x_train)
        test_pred = knn.predict(x_test)
        
        train_acc = metrics.accuracy_score(train_pred, y_train)
        test_acc = metrics.accuracy_score(test_pred, y_test)
        plot_data_train[int(k)] = float(test_acc)
        
    
    plot_lists = sorted(plot_data_train.items()) # sorted by key, return a list of tuples
    plot_x, plot_y = zip(*plot_lists) # unpack a list of pairs into two tuples
    
    sns.pointplot(x=list(plot_x), y=list(plot_y))
    plt.title('Different K Values by Accuracy')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Value of K')
    plt.show()
    
    optimal_k, optimal_score = max(plot_data_train.items(), key=operator.itemgetter(1))
    print(' Most \'Accuracy\' K: {} \n Associated Accuracy Score: {}'.format(optimal_k, optimal_score))
    return optimal_k


def confusion_matrix(y_actual, y_predict):
    '''
    Build confusion matrix based on actual & predicted values.
    Inputs: 
    - y_actual (Series): Actual predictor values from original dataset
    - y_predict (Series): Predicted values produced by classifier
    Returns a confusion matrix
    '''
    return metrics.confusion_matrix(y_actual, y_predict, labels=None, sample_weight=None)


def compare_dtrees(x_train, y_train, x_test, y_test, depths=[1, 2, 3, 4, 5]):
    '''
    Similar to the test_different_k function, but for 
    '''
    depths = [1, 2, 3, 4, 5]
    plot_data_train = dict()
    for d in depths:
        dec_tree = DecisionTreeClassifier(max_depth=d)
        dec_tree.fit(x_train, y_train)
        train_pred = dec_tree.predict(x_train)
        test_pred = dec_tree.predict(x_test)
 
        train_acc = metrics.accuracy_score(train_pred, y_train)
        test_acc = metrics.accuracy_score(test_pred, y_test)
        
        print("Depth: {} – Train acc: {:.4f} | Test acc: {:.4f}".format(d, train_acc, test_acc))
        
        plot_data_train[int(d)] = float(test_acc)
        
    plot_lists = sorted(plot_data_train.items()) # sorted by key, return a list of tuples
    plot_x, plot_y = zip(*plot_lists) # unpack a list of pairs into two tuples
    
    sns.pointplot(x=list(plot_x), y=list(plot_y))
    plt.title('Different d Values by Accuracy')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Depth')


def simple_logit(df, x_train, y_train, features=None, outcome_col=None):
    '''
    build a logit model.
        
    input 
       feature_list (list of colnames): the list of features to include
       y_column (str colname): the column name for what you want to predict
        
    returns: model (sklearn's logistic regression model)
    '''
    # this ensures that the data is scaled
    model = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',linear_model.LogisticRegressionCV())
    ])
    # this fits the model
    model = model.fit(data[feature_list], data[y_column])
    return model


def validate_clf(x, y, y_column, clf):
    '''
    prints: 
        Accuracy score
        classification report (precision, recall, f1-score, and support (n))
        ROC-AUC score
    
    input:
        df
        features
        y_column - outcome variable
        clf to eval
    '''
    y_pred=clf.predict(x)   
    print ("ACCURACY SCORE: {:.4f}".format(metrics.accuracy_score(y,y_pred)))
    print ("CLASSIFICATION REPORT\n", metrics.classification_report(y,y_pred))
    probs = clf.predict_proba(x)
    print ("ROC-AUC: {:.3f}".format(roc_auc_score(y, probs[:, [1]])), "\n")

NOTEBOOK=1

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="log", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['log'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['log'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['log'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

    
    
def clf_loop(models_to_run, clfs, grid, X, y):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df


def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
    

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def unstack_vars_juniorhigh(df, pivot_index='student_lookup', pivot_col='grade', pivot_var='', aggfunc = None):
    '''
    This function takes a long-form dataset, such as the results of an SQL query over multiple school years, 
    and unpacks one variable over 6th, 7th, and 8th grades. 
    
    This ONLY works for numeric columns.
    '''
    if aggfunc is not None:
        pivot_table = df.pivot_table(index=[pivot_index], columns=[pivot_col], values=[pivot_var], aggfunc=aggfunc)
    else:
        pivot_table = df.pivot_table(index=[pivot_index], columns=[pivot_col], values=[pivot_var])

    sixth = pivot_table.unstack()[pivot_var][6]
    seventh = pivot_table.unstack()[pivot_var][7]
    eighth = pivot_table.unstack()[pivot_var][8]
    
    if len(sixth.values) == 0:
        seventh = pd.DataFrame(seventh)
        eighth = pd.DataFrame(eighth)
        df = seventh.join(eighth, lsuffix='07_'+str(pivot_var), rsuffix='08_'+str(pivot_var))
        return df
    else:
        sixth = pd.DataFrame(sixth)
        seventh = pd.DataFrame(seventh)
        eighth = pd.DataFrame(eighth)

    
    df_temp = sixth.join(seventh, lsuffix='06_'+str(pivot_var), rsuffix='07_'+str(pivot_var))
    df = df_temp.join(eighth, rsuffix='08_'+str(pivot_var))
    # For some reason I am getting an extra column with '0' as the title.
    df['008_'+str(pivot_var)] = df[0]
    df.drop(columns=0, inplace=True)
    return df


def combine_multiple_vars(df, pivot_index='student_lookup', pivot_col='grade', variable_list='', aggfunc=None):
    '''
    This function takes a list of variables to unpack, and calls unstack_vars_juniorhigh on each of them. 
    The resulting dataframes are then joined together and returned.
    '''
    df_holder = []
    for var in variable_list:
        if aggfunc is not None:
            temp = unstack_vars_juniorhigh(df, pivot_index, pivot_col, var, aggfunc)
        else:
            temp = unstack_vars_juniorhigh(df, pivot_index, pivot_col, var)
        df_holder.append(temp)

    initial = df_holder[0] 
    final = None
    for df in df_holder[1:]:
        if final is not None:
            final = final.join(df)
        else:
            final = initial.join(df)
    return final

def collapse_categoricals(df, new_var, list_of_vars):
    '''
    Collapses 6th, 7th, and 8th grade binary variables into one, then drops the separate columns.
    '''
    sixth_grade = list_of_vars[0]
    seventh_grade = list_of_vars[1]
    eighth_grade = list_of_vars[2]
    df[new_var] = df[sixth_grade].combine_first(df[seventh_grade])
    df[new_var] = df[new_var].combine_first(df[eighth_grade])

    df.drop(list_of_vars, axis=1, inplace=True)
    return df

def plot_precision_recall(model_dict=None, color_list=None):
    for time in model_dict.keys():
        handles = []
        fig, ax = plt.subplots()

        for i, model in enumerate(model_dict[time].keys()):
            if model != 'svm':
                cmap=colors.ListedColormap(colors=color_list[i])
                yp, yt = model_dict[time][model]['y_pred'], model_dict[time][model]['y_test']
                t = 'Precisision_recall for ' + str(time)
                skplt.metrics.plot_precision_recall(yt, yp, classes_to_plot=[1], 
                                            plot_micro=False, cmap=cmap, ax=ax, title=t) 
                handles.append(lines.Line2D([], [], color=color_list[i], label=model))  
        plt.legend(handles=handles)
        
        
        
        
        
        
        