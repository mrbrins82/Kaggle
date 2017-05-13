from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier as XGB

def generate_data_frame():
    """
        """
    t0 = time.time()
    print 'Generating data frame...'

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # drop columns that don't have much affect on outcome
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # turn gender string into an integer using a new column titled
    # 'Gender' and drop the 'Sex' column
    train_df['Gender'] = train_df.Sex.map({'female': 0, 'male': 1}).astype(int)
    train_df = train_df.drop(['Sex'], axis=1)
    test_df['Gender'] = test_df.Sex.map({'female': 0, 'male': 1}).astype(int)
    test_df = test_df.drop(['Sex'], axis=1)

    # passengers 61 and 829 don't have 'Embarked' values
    # they are both 1st class women, from the data the mean class of women
    # embarking at the different ports are
    # 'C' ---> female_mean_class = 1.726
    # 'Q' ---> female_mean_class = 2.889
    # 'S' ---> female_mean_class = 2.197
    # it seems reasonable to guess that these passengers embarked at Cherbourg
    # the test data is complete in the 'Embarked' column

    train_df.loc[ (train_df.Embarked.isnull()), 'Embarked'] = 'C'
    train_df.Embarked = train_df.Embarked.map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
    test_df.Embarked = test_df.Embarked.map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    # some passenger ages are missing so let's duplicate the 'Age' column with
    # an 'AgeFill' column and then populate the missing values 
    train_df['AgeFill'] = train_df.Age
    test_df['AgeFill'] = test_df.Age

    mean_ages = np.zeros((2, 3, 3))
    for ii in range(2):
        for jj in range(3):
            for kk in range(3):
                mean_ages[ii, jj, kk] = train_df[ (train_df['Gender'] == ii) & (train_df['Pclass'] == jj+1) \
                                                & (train_df['Embarked'] == kk)]['Age'].dropna().mean()

    for ii in range(2):
        for jj in range(3):
            for kk in range(3):
                train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == ii) & (train_df.Pclass == jj+1) \
                            & (train_df.Embarked == kk), 'AgeFill'] = mean_ages[ii, jj, kk]
                test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == ii) & (test_df.Pclass == jj+1) \
                            & (test_df.Embarked == kk), 'AgeFill'] = mean_ages[ii, jj, kk]

    train_df = train_df.drop(['Age'], axis=1)
    test_df = test_df.drop(['Age'], axis=1)

    # It could be that one's gender and class couple together in their effects so we will
    # add new column 'GenderClass' which takes the product of passenger gender value (plus one) and class
    # examples: a first class female would have GenderClass = (0 + 1) * 1 = 1
    #           a second class male would have GenderClass = (1 + 1) * 2 = 4
#    train_df['GenderClass'] = (train_df.Gender + 1) * train_df.Pclass
#    test_df['GenderClass'] = (test_df.Gender + 1) * test_df.Pclass

    # Family size could play a role in survival rate as well
#    train_df['FamilySize'] = train_df.SibSp + train_df.Parch
#    test_df['FamilySize'] = test_df.SibSp + test_df.Parch

    # One's gender may also couple to family size when determining survival as well just as
    # gender and class (again it will be gender + 1 as was in GenderClass
    # we also take family size + 1 in order to distinguish men and women that are alone)

#    train_df['GenderFamilySize'] = (train_df.Gender + 1) * (train_df.FamilySize + 1)
#    test_df['GenderFamilySize'] = (test_df.Gender + 1) * (test_df.FamilySize + 1)

    # passenger 1044 is a 3rd class male 60.5yrs old who embarked at Southampton
    # and he is missing his fare value, from data the mean fare for 3rd class 
    # man embarking at Southampton is 13.307149

    test_df.loc[ (test_df.Fare.isnull()), 'Fare'] = 13.307149

    train_df = train_df.drop(['SibSp', 'Parch', 'Embarked'], axis=1)
    test_df = test_df.drop(['SibSp', 'Parch', 'Embarked'], axis=1)

    print 'Finished generating data frame.'
    print 'function to generate data frame took %.1fs\n'%(time.time() - t0)

    return train_df, test_df

def split_data(df):
    """
        splits training data for cross validation

        """

    X = df.drop(['Survived'], axis=1)
    y = df.Survived

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    T = time.time()
    
    # load the prepared data frames
    train_df, test_df = generate_data_frame()
    final_train_df = train_df.drop(['Survived'], axis=1)
    targets_df = train_df.Survived

    # split the training data frame into parts for cross validation
    X_train, X_test, y_train, y_test = split_data(train_df)

    # create a pipeline for grid search
    xgb_args = {'nthread':2, 'subsample':0.8, 'seed':321}

    pipeline = Pipeline([
                        ('xgb', XGB(**xgb_args))
                        ])

    parameters = {
                  'xgb__n_estimators': (100, 150, 200, 250, 300),
                  'xgb__max_depth': np.arange(1, 10, 1),
                  'xgb__learning_rate': np.arange(0.01, 0.2, 0.01), 
                  'xgb__colsample_bytree': np.arange(0.5, 0.96, 0.05)
                  }

    # begin the grid search
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f'%grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' %(param_name, best_parameters[param_name])
        if param_name == 'clf__min_samples_split':
            min_samples_split = best_parameters[param_name]
        elif param_name == 'clf__n_estimators':
            n_estimators = best_parameters[param_name]
#        elif param_name == 'pca__n_components':
#            n_components = best_parameters[param_name]
        elif param_name == 'clf__max_depth':
            max_depth = best_parameters[param_name]


    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)

    for param_name in parameters.keys():
        xgb_args[param_name[5:]] = best_parameters[param_name]

    classifier = XGB(**xgb_args)
    classifier.fit(final_train_df, targets_df)
    output = classifier.predict(test_df)

    PassengerIds = np.arange(892, 1310)
    S = Series(output, index=PassengerIds, dtype=int)
    S.to_csv('Titanic_xgboost_results.csv', header=True, index_label=['PassengerId', 'Survived'])

