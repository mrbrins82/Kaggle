import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier


def load_data():

    # load the train/test sets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # drop ticket feature
    train_df = train_df.drop(['Ticket'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)

    ##########################
    # FILL IN MISSING VALUES #
    ##########################

    # embarked
    train_df.Embarked.iloc[61] = 'C'
    train_df.Embarked.iloc[829] = 'C'

    # fare
    test_df.Fare.iloc[152] = train_df[train_df.Pclass == 3].Fare.mean()

    ###############################
    # ENCODE CATEGORICAL FEATURES #
    ###############################

    # sex
    le_Sex = LabelEncoder()
    train_df.Sex = le_Sex.fit_transform(train_df.Sex)
    test_df.Sex = le_Sex.transform(test_df.Sex)

    # embarked
    le_Embarked = LabelEncoder()
    train_df.Embarked = le_Embarked.fit_transform(train_df.Embarked)
    test_df.Embarked = le_Embarked.transform(test_df.Embarked)

    # cabin
    train_df.Cabin.fillna(value='X', inplace=True)
    test_df.Cabin.fillna(value='X', inplace=True)

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


    train_df = pd.get_dummies(train_df, columns=['CabinLetter'],
                              prefix=['Cabin'])
    test_df = pd.get_dummies(test_df, columns=['CabinLetter'],
                             prefix=['Cabin'])

    test_df['Cabin_T'] = 0 # there are no T cabins in the test set

    # reorder the test columns to match the training columns
    ordered_cols = train_df.drop(['Survived'], axis=1).columns
    test_df = test_df.reindex(columns=ordered_cols)

#    def cabin_letter(x):
#        if str(x)[0] == 'A':
#            return 1
#        elif str(x)[0] == 'B':
#            return 2
#        elif str(x)[0] == 'C':
#            return 3
#        elif str(x)[0] == 'D':
#            return 4
#        elif str(x)[0] == 'E':
#            return 5
#        elif str(x)[0] == 'F':
#            return 6
#        elif str(x)[0] == 'G':
#            return 7
#        elif str(x)[0] == 'T':
#            return 8
#        else:
#            return 9
#
#    train_df['CabinLetter'] = train_df.Cabin.apply(cabin_letter)
#    test_df['CabinLetter'] = test_df.Cabin.apply(cabin_letter)

    train_df = train_df.drop(['Cabin'], axis=1)
    test_df = test_df.drop(['Cabin'], axis=1)

    # name
    def get_title(x):
        title = str(x).split(',')[1].lstrip().split(' ')[0]
        return title

    train_df['Title'] = train_df.Name.apply(get_title)
    test_df['Title'] = test_df.Name.apply(get_title)

    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)

    le_Title = LabelEncoder()

    all_titles = pd.concat((train_df.Title, test_df.Title))
    le_Title.fit(all_titles)
    train_df['NumTitle'] = le_Title.transform(train_df.Title)
    test_df['NumTitle'] = le_Title.transform(test_df.Title)

    ###########################
    # FILL MISSING AGE VALUES #
    ###########################
    train_without_ages = train_df[train_df.Age.isnull()]
    test_without_ages = test_df[test_df.Age.isnull()]

    train_with_ages = train_df[train_df.Age.notnull()] # only use training data to fill ages

    def get_mean_age(pclass, sibsp, parch, numtitle):#, cabinletter):
        # we need to make sure that the passenger with missing age
        # info is not a completely unique passenger, otherwise there
        # there will be no other passengers from which to calculate
        # a mean age, so we will update the mean age as we reduce
        # the temp_df
        temp_df = train_with_ages[train_with_ages.Pclass == pclass]
        if temp_df.shape[0] != 0:
            mean_age = temp_df.Age.mean()

#        temp_df = temp_df[temp_df.CabinLetter == cabinletter]
#        if temp_df.shape[0] != 0:
#            mean_age = temp_df.Age.mean()

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
#        cabinletter = train_df.CabinLetter.iloc[Id - 1]
        numtitle = train_df.NumTitle.iloc[Id - 1]

        train_df.Age.iloc[Id - 1] = get_mean_age(pclass, sibsp, parch, numtitle)#, cabinletter)

    # fill in the test_df missing ages
    for Id in test_without_ages.PassengerId:    
        pclass = test_df.Pclass.iloc[Id - 892]
        sibsp = test_df.SibSp.iloc[Id - 892]
        parch = test_df.Parch.iloc[Id - 892]
#        cabinletter = test_df.CabinLetter.iloc[Id - 892]
        numtitle = test_df.NumTitle.iloc[Id - 892]

        test_df.Age.iloc[Id - 892] = get_mean_age(pclass, sibsp, parch, numtitle)#, cabinletter)

    #######################
    # FEATURE ENGINEERING #
    #######################

    # age
    def is_child(x):
        if int(x) <= 10:
            return 1
        else:
            return 0

    train_df['Child'] = train_df.Age.apply(is_child)
    test_df['Child'] = test_df.Age.apply(is_child)

    # sibsp & parch
    train_df['FamSize'] = train_df.SibSp + train_df.Parch
    test_df['FamSize'] = test_df.SibSp + test_df.Parch

    def is_alone(x):
        if int(x) == 0:
            return 1
        else:
            return 0

    train_df['Alone'] = train_df.FamSize.apply(is_alone)
    test_df['Alone'] = test_df.FamSize.apply(is_alone)

    def is_small_fam(x):
        if int(x) == 0:
            return 0
        elif int(x) >= 4:
            return 0
        else:
            return 1

    train_df['SmallFam'] = train_df.FamSize.apply(is_small_fam)
    test_df['SmallFam'] = test_df.FamSize.apply(is_small_fam)

    def is_large_fam(x):
        if int(x) >= 4:
            return 1
        else:
            return 0

    train_df['LargeFam'] = train_df.FamSize.apply(is_large_fam)
    test_df['LargeFam'] = test_df.FamSize.apply(is_large_fam)

    # fare
    def is_low_fare(x):
        if float(x) <= 10:
            return 1
        else:
            return 0

    train_df['LowFare'] = train_df.Fare.apply(is_low_fare)
    test_df['LowFare'] = test_df.Fare.apply(is_low_fare)

    # title
    def is_good_title(x):
        if str(x) == 'Mrs.':
            return 1
        elif str(x) == 'Miss.':
            return 1
        elif str(x) == 'Master.':
            return 1
        else:
            return 0

    train_df['GoodTitle'] = train_df.Title.apply(is_good_title)
    test_df['GoodTitle'] = test_df.Title.apply(is_good_title)

    train_df = train_df.drop(['Title'], axis=1)
    test_df = test_df.drop(['Title'], axis=1)

#    # cabin
#    def is_good_cabin(x):
#        if x == 'B':
#            return 1
#        elif x == 'C':
#            return 1
#        elif x == 'D':
#            return 1
#        elif x == 'E':
#            return 1
##        elif x == 'F':
##            return 1
#        else:
#            return 0
#
#    train_df['GoodCabin'] = train_df.CabinLetter.apply(is_good_cabin)
#    test_df['GoodCabin'] = test_df.CabinLetter.apply(is_good_cabin)
#
#    train_df = train_df.drop(['CabinLetter'], axis=1)
#    test_df = test_df.drop(['CabinLetter'], axis=1)

    return train_df, test_df


def main(train_df, test_df, cv, scoring, random_state):

    # we don't need the train_df passenger ids, but we will need
    # them from the test_df in order to write out the predictions
    # in a csv file
    train_df = train_df.drop(['PassengerId'], axis=1)

    test_Ids = test_df.PassengerId
    test_df = test_df.drop(['PassengerId'], axis=1)

    # split the training data and set aside a portion for validation
    X = train_df.drop(['Survived'], axis=1)
    Y = train_df.Survived

    np.random.seed(random_state) # so that split is reproducible
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    ####################
    # BUILD CLASSIFIER #
    ####################

    # gbclassifier
    gbc_params = {'random_state':np.array([random_state]),
                  'n_estimators':np.array([100, 150, 200, 225, 250]),
                  'max_features':['auto']
                   }

    gbc = GradientBoostingClassifier()
    
    # start the grid search
    gbc_grid_search = GridSearchCV(gbc, gbc_params, n_jobs=-1, 
                                   verbose=1, scoring=scoring, cv=cv)

    gbc_grid_search.fit(x_train, y_train)

    print 'Gradient Boosting Classifier'
    print '****************************'
    print 'Best %s score: %0.3f'%(scoring, gbc_grid_search.best_score_)
    print 'Best parameters:'
    gbc_best_parameters = gbc_grid_search.best_estimator_.get_params()
    for param_name in sorted(gbc_params.keys()):
        print '\t%s: %r'%(param_name, gbc_best_parameters[param_name])
    print ''

    best_gbc = gbc_grid_search.best_estimator_



#    # rfclassifier
#    rfc_params = {'random_state':np.array([random_state]),
#                  'n_jobs':np.array([-1]),
#                  'class_weight':['balanced'],
#                  'n_estimators':np.array([100, 150, 200, 500]),
#                  'max_features':np.array([5, 6, 7, 8]),
#                  'bootstrap':[True]
#                   }
#
#    rfc = RandomForestClassifier()
#    
#    # start the grid search
#    rfc_grid_search = GridSearchCV(rfc, rfc_params, n_jobs=-1, 
#                                   verbose=1, scoring=scoring, cv=cv)
#
#    rfc_grid_search.fit(x_train, y_train)
#
#    print 'Random Forest'
#    print '*************'
#    print 'Best %s score: %0.3f'%(scoring, rfc_grid_search.best_score_)
#    print 'Best parameters:'
#    rfc_best_parameters = rfc_grid_search.best_estimator_.get_params()
#    for param_name in sorted(rfc_params.keys()):
#        print '\t%s: %r'%(param_name, rfc_best_parameters[param_name])
#    print ''
#
#    best_rfc = rfc_grid_search.best_estimator_


    # xgbclassifier
    xgb_params = {'seed':np.array([random_state]),
                  'objective':['binary:logistic'],
                  'scale_pos_weight':np.array([1.605]),
                  'max_delta_step':np.array([0]),
                  'n_estimators':np.array([100, 150, 200]),
                  'learning_rate':np.logspace(-3, -1, 3),
                  'max_depth':np.array([3, 4, 5]),
#                  'gamma':np.array([0.0, 0.01]),
                  'reg_lambda':np.array([0.01, 0.1]),
#                  'subsample':np.array([0.5, 1.])
                  }

    xgb = XGBClassifier()

    # start the grid search
    xgb_grid_search = GridSearchCV(xgb, xgb_params, n_jobs=-1,
                                   verbose=1, scoring=scoring, cv=cv)
    xgb_grid_search.fit(x_train, y_train)

    print 'XGBClassifier'
    print '*************'
    print 'Best %s score: %0.3f'%(scoring, xgb_grid_search.best_score_)
    print 'Best parameters:'
    xgb_best_parameters = xgb_grid_search.best_estimator_.get_params()
    for param_name in sorted(xgb_params.keys()):
        print '\t%s: %r'%(param_name, xgb_best_parameters[param_name])
    print ''

    best_xgb = xgb_grid_search.best_estimator_

#    predictions_test = best_xgb.predict(x_test)

    vclf = VotingClassifier(estimators=[('xgb', best_xgb), ('gbc', best_gbc)], 
                            voting='soft', n_jobs=-1)
    vclf.fit(x_train, y_train)
    
    voting_acc = vclf.score(x_test, y_test)
    print 'Voting score: ',voting_acc

#    print '***************************'
#    print 'Random State    : %s'%str(random_state).zfill(6)
#    print 'GridSearch score: %.4f (%s)'%(grid_search.best_score_, scoring)
#    print 'accuracy score  : %f'%accuracy_score(y_test, predictions_test)
#    print 'precision score : %f'%precision_score(y_test, predictions_test)
#    print 'recall score    : %f'%recall_score(y_test, predictions_test)
#    print 'f1 score        : %f'%f1_score(y_test, predictions_test)
#    print '***************************'

#    predictions = best_xgb.predict(test_df)
    predictions = vclf.predict(test_df)
    predictions_df = pd.DataFrame(columns=['PassengerId', 'Survived'], data=zip(test_Ids, predictions))
        
    predictions_df.to_csv('./SubmissionFiles/cv' + str(cv) + '_rs' + str(random_state).zfill(6) + '_' + scoring + '.csv', index=False)

    return voting_acc

def generate_plot(cv, ii, final_prediction_acc):
    test_plot_dir = '/Users/matthewbrinson/Work/DataScience/Kaggle/Titanic/Plots/Testing/'

    if ii+1 > 1:
        fig = plt.figure(figsize=(12,6))

        ax1 = fig.add_subplot(111)
        sns.distplot(np.array(final_prediction_acc), bins=101, norm_hist=True, ax=ax1, color='red', label='Voting Pred.')
        ax1.set_xlim(xmin=.720, xmax=0.925)
        ax1.set_ylim(ymax=70)
        ax1.set_xlabel('accuracy score', size=18)
        ax1.legend(loc='upper left')

        plt.title('CV=%d, N=%d'%(cv, (ii+1)), size=20)
        plt.savefig(test_plot_dir + 'cv%d_acc_scores_hist_voting.png'%(cv))
        plt.clf()
        os.system('open ' + test_plot_dir + 'cv%d_acc_scores_hist_voting.png'%(cv))

if __name__ == '__main__':

    ###############################
    cv = 10
    scoring = 'accuracy'
    n_iter = 25
    make_plot = True

    ################################
    train_df, test_df = load_data()

#    train_df = train_df.drop(['FamSize', 'Parch', 'SibSp'], axis=1)
#    test_df = test_df.drop(['FamSize', 'Parch', 'SibSp'], axis=1)

    train_df = train_df[train_df.SibSp < 8]
#    train_df = train_df[train_df.Fare < 500]


    final_prediction_acc = []
    for ii in xrange(n_iter):
    
        random_state = np.random.randint(1, 100001)
        print ''
        print '****************'
        print '*  RS: %s  *'%str(random_state).zfill(6)
        print '****************'
        voting_acc = main(train_df, test_df, cv, scoring, random_state)


        if make_plot:

            final_prediction_acc.append(voting_acc)
            generate_plot(cv, ii, final_prediction_acc)


