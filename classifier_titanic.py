import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def load_data():
    
    # Load the training and testing data sets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # drop the 'Ticket' feature
    train_df = train_df.drop(['Ticket'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)

    ##########################
    # FILL IN MISSING VALUES #
    ##########################
    
    ##### EMBARKED #####
    train_df.Embarked.iloc[61] = 'C'
    train_df.Embarked.iloc[829] = 'C'

    ##### FARE #####
    test_df.Fare.iloc[152] = train_df[train_df.Pclass == 3].Fare.mean()

    ##### CABIN #####
    train_df.Cabin.fillna(value='X', inplace=True)
    test_df.Cabin.fillna(value='X', inplace=True)


    ###############################
    # ENCODE CATEGORICAL FEATURES #
    ###############################

    ##### SEX #####
    le_Sex = LabelEncoder()
    train_df.Sex = le_Sex.fit_transform(train_df.Sex)
    test_df.Sex = le_Sex.transform(test_df.Sex)

    ###### EMBARKED #####
    le_Embarked = LabelEncoder()
    train_df.Embarked = le_Embarked.fit_transform(train_df.Embarked)
    test_df.Embarked = le_Embarked.transform(test_df.Embarked)


    ###################################
    #####   FEATURE ENGINEERING   #####
    ###################################

    ##### TITLE #####
    def get_title(x):
        title = str(x).split(',')[1].lstrip().split(' ')[0]
        return title

    train_df['Title'] = train_df.Name.apply(get_title)
    test_df['Title'] = test_df.Name.apply(get_title)
    
    # we don't need 'Name' anymore
    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)


    ##### GOODTITLE #####
    def is_good_title(x):
        if x == 'Mrs.':
            return 1
        elif x == 'Miss.':
            return 1
        elif x == 'Master.':
            return 1
        else:
            return 0

    train_df['GoodTitle'] = train_df.Title.apply(is_good_title)
    test_df['GoodTitle'] = test_df.Title.apply(is_good_title)

    # encode the 'Title' feature
    le_Title = LabelEncoder()
    all_Titles = pd.concat((train_df.Title, test_df.Title))
    le_Title.fit(all_Titles)

    train_df['NumTitle'] = le_Title.transform(train_df.Title)
    test_df['NumTitle'] = le_Title.transform(test_df.Title)

    # we don't need 'Title' anymore
    train_df = train_df.drop(['Title'], axis=1)
    test_df = test_df.drop(['Title'], axis=1)


    #####   CABINLETTER   #####
    def get_cabin_letter(x):

        return str(x)[0]

    train_df['CabinLetter'] = train_df.Cabin.apply(get_cabin_letter)
    test_df['CabinLetter'] = test_df.Cabin.apply(get_cabin_letter)

    def is_good_cabin(x):
        if x == 'B':
            return 1
        elif x == 'C':
            return 1
        elif x == 'D':
            return 1
        elif x == 'E':
            return 1
        else:
            return 0

    train_df['GoodCabin'] = train_df.CabinLetter.apply(is_good_cabin)
    test_df['GoodCabin'] = test_df.CabinLetter.apply(is_good_cabin)

    # one-hot-encode the cabin letters
    train_df = pd.get_dummies(train_df, columns=['CabinLetter'], prefix=['Cabin'])

    test_df = pd.get_dummies(test_df, columns=['CabinLetter'],
                             prefix=['Cabin'])

    test_df['Cabin_T'] = 0 # there are no T cabins in the test set

    # reorder the test columns to match the training columns
    ordered_cols = train_df.drop(['Survived'], axis=1).columns
    test_df = test_df.reindex(columns=ordered_cols)
    
    # we don't need 'Cabin' anymore   
    train_df = train_df.drop(['Cabin'], axis=1)
    test_df = test_df.drop(['Cabin'], axis=1)


    #####   FAMSIZE   #####
    train_df['FamSize'] = train_df.SibSp + train_df.Parch
    test_df['FamSize'] = test_df.SibSp + test_df.Parch


    #####   ALONE   #####
    def is_alone(x):
        if x == 0:
            return 1
        else:
            return 0

    train_df['Alone'] = train_df.FamSize.apply(is_alone)
    test_df['Alone'] = test_df.FamSize.apply(is_alone)


    #####   SMALLFAM   #####
    def is_small_family(x):
        if x == 0:
            return 0
        elif x >= 4:
            return 0
        else:
            return 1

    train_df['SmallFam'] = train_df.FamSize.apply(is_small_family)
    test_df['SmallFam'] = test_df.FamSize.apply(is_small_family)


    #####   LARGEFAM   #####
    def is_large_family(x):
        if x >= 4:
            return 1
        else:
            return 0
    
    train_df['LargeFam'] = train_df.FamSize.apply(is_large_family)
    test_df['LargeFam'] = test_df.FamSize.apply(is_large_family)


    #####   LOWFARE   #####
    def is_low_fare(x):
        if x <= 10:
            return 1
        else:
            return 0

    train_df['LowFare'] = train_df.Fare.apply(is_low_fare)
    test_df['LowFare'] = test_df.Fare.apply(is_low_fare)


    return train_df, test_df


def fill_missing_ages(train_df, test_df):
    """
        Function that fills missing age values based on the 
        mean age for a given Pclass, SibSp, Parch, and NumTitle.
        Looking at the correlation between age and other 
        features, these seem to be the strongest.
    """
    
    ################################################################
    # We need to fill in the missing ages for 177 passengers in the 
    # train data set, and 86 passengers in the test data set. 
    ################################################################

    train_without_ages = train_df[train_df.Age.isnull()]
    test_without_ages = test_df[test_df.Age.isnull()]

    train_with_ages = train_df[train_df.Age.notnull()] # only use training data to fill ages

    def get_mean_age(pclass, sibsp, parch, numtitle):
        # we need to make sure that the passenger with missing age
        # info is not a completely unique passenger, otherwise there
        # there will be no other passengers from which to calculate
        # a mean age, so we will update the mean age as we reduce
        # the temp_df
        temp_df = train_with_ages[train_with_ages.Pclass == pclass]
        if temp_df.shape[0] != 0:
            mean_age = temp_df.Age.mean()

        temp_df = temp_df[temp_df.Parch == parch]
        if temp_df.shape[0] != 0:
            mean_age = temp_df.Age.mean()

        temp_df = temp_df[temp_df.NumTitle == numtitle]
        if temp_df.shape[0] != 0:
            mean_age = temp_df.Age.mean()

        temp_df = temp_df[temp_df.SibSp == sibsp]
        if temp_df.shape[0] != 0:
            mean_age = temp_df.Age.mean()

        return mean_age

    # fill in the train_df missing ages
    for Id in train_without_ages.PassengerId:        
        pclass = train_df.Pclass.iloc[Id - 1]
        sibsp = train_df.SibSp.iloc[Id - 1]
        parch = train_df.Parch.iloc[Id - 1]
        numtitle = train_df.NumTitle.iloc[Id - 1]

        train_df.Age.iloc[Id - 1] = get_mean_age(pclass, sibsp, parch, numtitle)

    # fill in the test_df missing ages
    for Id in test_without_ages.PassengerId:    
        pclass = test_df.Pclass.iloc[Id - 892]
        sibsp = test_df.SibSp.iloc[Id - 892]
        parch = test_df.Parch.iloc[Id - 892]
        numtitle = test_df.NumTitle.iloc[Id - 892]

        test_df.Age.iloc[Id - 892] = get_mean_age(pclass, sibsp, parch, numtitle)


    ##### CHILD #####

    # now that we have all ages filled, we can engineer one more feature
    # that is determined by a passenger being either younger or older than 10
    def is_child(x):
        if x <= 5:
            return 1
        else:
            return 0

    train_df['Child'] = train_df.Age.apply(is_child)
    test_df['Child'] = test_df.Age.apply(is_child)


    return train_df, test_df


def main(train_df, test_df, test_Ids, scoring, cv, train_split_frac, random_state):

    # split the data for cross validation
    X = train_df.drop(['Survived'], axis=1)
    Y = train_df.Survived
        
    np.random.seed(random_state) # so train_test_split is reproducible
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_split_frac)


    #################################################
    ### xgb classifier parameters for grid search ###
    #################################################
    fixed_params = {'seed':random_state,
                    'objective':'binary:logistic',
                    'scale_pos_weight':1.605,
                   }

    xgb = XGBClassifier(**fixed_params)

    test_params = {'n_estimators':np.array([50, 100, 150, 200]),
                   'learning_rate':np.logspace(-3, -1, 3),
                   'max_depth':np.array([3, 4, 5, 6]),
                   'gamma':np.array([0., 0.1]),
                   'max_delta_step':np.array([0., 0.001]),
                   'reg_lambda':np.array([0.01, 0.1])
                  }

    grid_search = GridSearchCV(xgb, test_params, n_jobs=-1, verbose=1, scoring=scoring, cv=cv)
    grid_search.fit(x_train, y_train)
    
    # print out grid search best score and parameters
    print 'Best %s score: %0.3f'%(scoring, grid_search.best_score_)
    print 'Best Parameters:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])

    # check how classifier performs of the rest of the training set
    predictions_test = grid_search.predict(x_test)

    predictions = grid_search.predict(test_df)
    prediction_df = pd.DataFrame(columns=['PassengerId', 'Survived'], data=zip(test_Ids, predictions))
    prediction_df.to_csv('./submission_titanic.csv', index=False)



if __name__ == '__main__':

    scoring = 'accuracy' # evaluation metric for grid search
    cv = 3 # number of cross validation sets
    train_split_frac = 3/4. # fraction of training set to use for training, the rest will go to validation

    train_df, test_df = load_data()
    train_df, test_df = fill_missing_ages(train_df, test_df) # fill in missing ages

    # we don't need the passenger IDs until the end
    train_df = train_df.drop(['PassengerId'], axis=1)
    test_Ids = test_df.PassengerId
    test_df = test_df.drop(['PassengerId'], axis=1)

        
    random_state = np.random.randint(1, 100001)
    main(train_df, test_df, test_Ids, scoring, cv, train_split_frac, random_state)


