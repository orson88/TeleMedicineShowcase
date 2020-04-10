import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_data(file):
    '''
    Reads the data from the file and preprocess it
    '''
    data = pd.read_csv(file)    ## reading data from the file
    y = data['Survived']        ## creating answers on training data
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}) ## data proccesing
    data = data.drop(['Survived', 'Name'], axis = 1)
    data = data.fillna(value = 0)   ## fill NaN with 0
    data = pd.get_dummies(data) ## create dummy feuters since sklearn works only with numbers
    return data, y

def best_decision_tree():
    '''
    Create the best decision tree using the Cross Validation
    '''
    tree = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, \
            max_features = 231, random_state = 11)
    # params = {
    #     'max_depth': np.arange(2, 12),
    #     'min_samples_split': np.arange(2, 12),            ## have already implemented and found the optimal params
    #     'max_features': np.arange(220, 241)
    # }
    # best_tree = GridSearchCV(tree, param_grid = params, n_jobs = -1, verbose = True)
    return tree

def best_random_forest():
    '''
    Create the best random forest using the Cross Validation
    '''
    forest = RandomForestClassifier(n_estimators = 28, oob_score = True, n_jobs = -1,\
                 verbose = True, random_state = 11, max_depth = 11, min_samples_split = 6, max_features = 290)
    ## the best params for this tree were found previoulsy with GridSearchCV
    return forest


def main():
    file = "train.csv"
    data, y = read_data(file)   ## reading the training data and answers
    ## splitting data to train set and test set
    X_train, X_valid, y_train, y_valid = train_test_split(data, y, test_size = 0.3, random_state = 11)
    del data

    decision_tree = best_decision_tree()    ## creating decision tree
    decision_tree.fit(X_train, y_train)     ## fitting decision tree
    ## accuracy_score == 0.8208955223880597

    random_forest = best_random_forest()    ## creating random forest
    random_forest.fit(X_train, y_train)     ## fitting random forest
    ## accuracy_score == 0.8432835820895522
    print(accuracy_score(random_forest.predict(X_valid), y_valid))

    logit = LogisticRegression(random_state = 11, C = 1, \
                verbose = True, max_iter = 71)    ## creating Logistic Regression
    logit.fit(X_train, y_train)     ## fitting Logistic Regression
    ## accuracy_score == 0.832089552238806


if __name__ == '__main__':
    main()
