from __future__ import division

import numpy as np
import pandas as pd
import time

from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from xgboost import XGBClassifier

def generate_frames():
    """
        """
    
    t0 = time.time()
    print 'Generating data frames...'

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    drop_list = []
    for col in train_df.columns:
        if train_df[col].std() == 0:
            drop_list.append(col)

    train_df = train_df.drop(drop_list, axis=1)
    test_df = test_df.drop(drop_list, axis=1)

    drop_list = []
    c = train_df.columns
    for ii in range(len(c) - 1):
        v = train_df[c[ii]].values
        for jj in range(ii+1, len(c)):
            if np.array_equal(v, train_df[c[jj]].values):
                drop_list.append(c[jj])

    train_df = train_df.drop(drop_list, axis=1)
    test_df = test_df.drop(drop_list, axis=1)

    X = train_df.drop(['TARGET'], axis=1)
    y = train_df.TARGET

    print 'Finished generating data frames in: %.2fs\n'%(time.time() - t0)
    
    return train_df, test_df, X, y

if __name__ == '__main__':

    train_df, test_df, X, y = generate_frames()

    final_train_df = train_df.drop(['TARGET', 'ID'], axis=1)
    final_targets_df = train_df.TARGET

    Ids = test_df.ID
    final_test_df = test_df.drop(['ID'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_scaler = preprocessing.StandardScaler()
    scaled_X_train = X_scaler.fit_transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    xgb_args = {'nthread':2, 'seed':321, 'n_estimators':350, 'subsample':0.9, 'colsample_bytree':0.8, 'learning_rate':0.03}

    pipeline = Pipeline([
                        ('xgb', XGBClassifier(**xgb_args))
                        ])

    parameters = {
                   'xgb__max_depth': np.arange(3, 6, 1),
#                   'xgb__n_estimators': (300, 350)
#                   'xgb__learning_rate': (0.01, 0.03, 0.05),
#                   'xgb__colsample_bytree': (0.8, 0.85)
                 }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, scoring='roc_auc', cv=8)

    # the grid search used to contain the scaled values
    grid_search.fit(X_train, y_train)
    print 'Best score: %.3f'%grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' %(param_name, best_parameters[param_name])

    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)

    for param_name in parameters.keys():
        xgb_args[param_name[5:]] = best_parameters[param_name]

    print 'xgb_args:', xgb_args

    final_scaler = preprocessing.StandardScaler()
    scaled_final_train_df = final_scaler.fit_transform(final_train_df)
    scaled_final_test_df = final_scaler.transform(final_test_df)

    # the lines below used to contain the scaled values
    classifier = XGBClassifier(**xgb_args)
    classifier.fit(final_train_df, final_targets_df)
    output = classifier.predict_proba(final_test_df)[:,1]

    S = Series(output, index=Ids)
    S.to_csv('Santander_xgboost_results_3.csv', header=True, index_label=['ID', 'TARGET'])

