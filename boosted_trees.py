import pandas as pd
import numpy as np
import datetime

from dateutil import parser
from pandas import DataFrame
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier as XGB

seed=321

def load_data():
    """
        this function loads the train and test
        data sets, then removes several columns
        and replaces interest level with a 
        numerical value, we also need the 
        listing IDs for the submission file
    """
    # load the data
    print 'Loading training/test data...'
    train_df = pd.read_json('train.json')
    test_df = pd.read_json('test.json')
    listing_id = test_df.listing_id

    # dropping columns that probably won't have much effect
    # may end up putting some of these back later, i.e. the
    # 'description' could be used later looking at sentiment
    # analysis
    train_df = train_df.drop(['description', 'display_address', 'manager_id', 'street_address'], axis=1)
    test_df = test_df.drop(['description', 'display_address', 'manager_id', 'street_address'], axis=1)

    # replace 'interest_level' with numerical values
    train_df['interest_level'].replace(to_replace=['low', 'medium', 'high'], value=[-1, 0, 1], inplace=True)

    # the day of year and day of month have some decent correlation with interest level
    map_to_day_of_year = lambda x: int(x.strftime('%j'))
    map_to_day_of_month = lambda x: int(x.day)

    train_df.created = pd.to_datetime(train_df.created)
    test_df.created = pd.to_datetime(test_df.created)
    train_df['day_of_year'] = train_df['created'].apply(map_to_day_of_year)
    train_df['day_of_month'] = train_df['created'].apply(map_to_day_of_month)
    test_df['day_of_year'] = test_df['created'].apply(map_to_day_of_year)
    test_df['day_of_month'] = test_df['created'].apply(map_to_day_of_month)

    # there is also some correlation with the number of features and interest level
    # as well as the number of photos in the listing
    map_to_number_of_features = lambda x: len(x)
    train_df['number_of_features'] = train_df['features'].apply(map_to_number_of_features)
    train_df['number_of_photos'] = train_df['photos'].apply(map_to_number_of_features)
    test_df['number_of_features'] = test_df['features'].apply(map_to_number_of_features)
    test_df['number_of_photos'] = test_df['photos'].apply(map_to_number_of_features)
    
    # there's also a pretty good correlation between listings that have a building_id
    # and interest level as well
    def conditional(x):
        if x == unicode(0):
            return 0
        else:
            return 1

    map_for_building_id = lambda x: conditional(x)
    train_df['building_id'] = train_df['building_id'].apply(map_for_building_id)
    test_df['building_id'] = test_df['building_id'].apply(map_for_building_id)

    # now we drop the 'features', 'photos', 'building_id', and 'created' features
    train_df = train_df.drop(['features', 'building_id', 'created', 'photos'], axis=1)
    test_df = test_df.drop(['features', 'building_id', 'created', 'photos'], axis=1)

    return train_df, test_df, listing_id

def split_data(data):
    """
        this function splits the training data set
        for cross validation
    """
    print 'Splitting data for cross validation...'

    X = data.drop(['interest_level'], axis=1)
    Y = data.interest_level

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    return X_train, X_test, Y_train, Y_test

def perform_grid_search(x_train, x_test, y_train, y_test):
    """
        this is the function that performs the grid search
        and prints out the classification report with best 
        parameters of our grid search
    """
    print 'Performing the grid search...'
    pipeline = Pipeline([
                        ('xgb', XGB(objective='multi:softprob', nthread=2, subsample=0.75, seed=seed))
                        ])
    parameters = {
                    'xgb__n_estimators': np.array([250]),
                    'xgb__colsample_bytree': np.array([0.7, 0.8]),
                    'xgb__max_depth': np.array([3, 4, 5]),
                    'xgb__learning_rate': np.array([0.1, 0.5]),
                    'xgb__gamma': np.array([0.01, 0.1]),
                    'xgb__scale_pos_weight': np.array([0.1, 1.0, 5.]),
                    'xgb__max_delta_step': np.array([5, 10])
                 }
   
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, scoring='log_loss', cv=3)
    grid_search.fit(x_train, y_train)
    print 'Best score: %0.3f'%grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' %(param_name, best_parameters[param_name])
        if param_name == 'xgb__colsample_bytree':
            colsample_bytree = best_parameters[param_name]
        elif param_name == 'xgb__n_estimators':
            n_estimators = best_parameters[param_name]
        elif param_name == 'xgb__max_depth':
            max_depth = best_parameters[param_name]
        elif param_name == 'xgb__learning_rate':
            learning_rate = best_parameters[param_name]
        elif param_name == 'xgb__gamma':
            gamma = best_parameters[param_name]
        elif param_name == 'xgb__scale_pos_weight':
            scale_pos_weight = best_parameters[param_name]
        elif param_name == 'xgb__max_delta_step':
            max_delta_step = best_parameters[param_name]

    predictions = grid_search.predict(x_test)
    print classification_report(y_test, predictions)

    return colsample_bytree, learning_rate, n_estimators, max_depth, gamma, scale_pos_weight, max_delta_step

def main():
    """
        main function
    """
    # load in data
    train_df, test_df, listing_id = load_data()
    print 'Finished loading data.'
    
    # split data for cross validation
    X_train, X_test, Y_train, Y_test = split_data(train_df)
    print 'Finished splitting data.'

    # perform the grid search
    colsample_bytree, learning_rate, n_estimators, max_depth, gamma, scale_pos_weight, max_delta_step = perform_grid_search(X_train, X_test, Y_train, Y_test)

    classifier = XGB(objective='multi:softprob', nthread=2, subsample=0.75, seed=seed, gamma=gamma, scale_pos_weight=scale_pos_weight, n_estimators=n_estimators, max_depth=max_depth, colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_delta_step=max_delta_step)
    
    # we need to split the train data and the target variable
    final_train_df = train_df.drop(['interest_level'], axis=1)
    targets_df = train_df.interest_level
    classifier.fit(final_train_df, targets_df)

    test_output = classifier.predict_proba(X_test)
    print 'log-loss = %.5f'%log_loss(Y_test, test_output, eps=1e-15)

#    check_output = classifier.predict(test_df)
#    print 'check ouput'
#    print check_output

    output = classifier.predict_proba(test_df)
    print 'output -->'
    print output

    # compute the log loss
#    log_loss = multi_class_log_loss(output, Y_test)
#    print 'log-loss = %0.5f'%log_loss

#print output

    D = DataFrame(output, index=listing_id, columns=['low', 'medium', 'high'])
#    print D
    
    # We need to interchange the low and high columns to match submission
    # format
    D_switched = D[['high', 'medium', 'low']]
#    print D_switched
    
    D_switched.to_csv('TwoSigma_BoostedTrees_results8.csv', header=True)

main()
