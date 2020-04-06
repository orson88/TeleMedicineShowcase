
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def stop_words():
    ## Returns the stop words that should be ignored while analyzing the data
    words = "которых которые твой которой которого сих ком свой твоя этими слишком нами всему будь саму чаще ваше сами наш затем еще самих наши ту каждое мочь весь этим наша своих оба который зато те этих вся ваш такая теми ею которая нередко каждая также чему собой самими нем вами ими откуда такие тому та очень сама нему алло оно этому кому тобой таки твоё каждые твои мой нею самим ваши ваша кем мои однако сразу свое ними всё неё тех хотя всем тобою тебе одной другие этао само эта буду самой моё своей такое всею будут своего кого свои мог нам особенно её самому наше кроме вообще вон мною никто это"
    words = words.split()
    return words

def read_data():
    ## Returns 4 datasets: X_train, X_valid, y_train, y_valid
    ## X_train, y_train - training data and target info
    ## X_valid, y_valid - test data: X_valid is info, y_valid is target for checking

    df = pd.read_csv('train_data.csv')  ## main dataframe
    df = df.drop('Unnamed: 0', axis = 1)
    y = df['score']     ## target
    y = y.map({'Позитивный': 1, 'Негативный': 0})   ## changing str to int
    df = df.drop('score', axis = 1)
    X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size = 0.4, random_state = 11) ## spliting the data
    return X_train, X_valid, y_train, y_valid

def main():
    logit = LogisticRegression(random_state = 11, solver = 'sag', n_jobs = 2, verbose = True)
    tf_idf = TfidfVectorizer (ngram_range = (1, 2), max_features=50000, stop_words = stop_words())
    logit_tfidf_pipeline = Pipeline([('tf_idf', tf_idf), ('logit', logit)], verbose = True)

    X_train, X_valid, y_train, y_valid = read_data()
    for column in X_train.columns:
        logit_tfidf_pipeline.fit(X_train[column], y_train)  ## training the model

    print(accuracy_score(logit_tfidf_pipeline.predict(X_valid['text']), y_valid))   ## checking the accuracy score

    while True:
         review = input("Enter your review: ")
         print(logit_tfidf_pipeline.predict_proba([review]))
         print(logit_tfidf_pipeline.predict([review]))

if __name__ == '__main__':
    main()
