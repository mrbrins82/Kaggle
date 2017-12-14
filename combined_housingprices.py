import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def load_data():
    """
        Loads training and testing data sets, as well
        as fills missing values and adding features.
    """
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')


    # There are some outliers in the training data
    train_df = train_df[train_df.GrLivArea < 4000]
    train_df = train_df[train_df.TotalBsmtSF < 4000]
    train_df = train_df[train_df['1stFlrSF'] < 4000]
    train_df = train_df[train_df.LotArea < 100000]
    train_df = train_df[train_df.LotFrontage < 200]

    
    # Create new column with boolean values based on cost of home types
    train_df['HighPriceTypes'] = train_df.MSSubClass.replace(to_replace=[20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190],
                                                                 value=[1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0])
    test_df['HighPriceTypes'] = test_df.MSSubClass.replace(to_replace=[20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190],
                                                                 value=[1,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0])


    # Create new column with mean home values for each neighborhood
    neighborhoods = train_df.Neighborhood.unique()
    neighborhood_means_dict = {neighborhood:train_df[train_df.Neighborhood == neighborhood].LotFrontage.mean()
                               for neighborhood in neighborhoods}

    neighborhood_means = np.array([neighborhood_means_dict[neighborhood] for neighborhood in neighborhoods])

    train_df['NeighborhoodMean'] = train_df.Neighborhood.replace(to_replace=neighborhoods, value=neighborhood_means)
    test_df['NeighborhoodMean'] = test_df.Neighborhood.replace(to_replace=neighborhoods, value=neighborhood_means)


    # Encode all of the categorical data into numerical data
    categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        le = LabelEncoder()
        combined_series = pd.concat((train_df[column], test_df[column]), axis=0)
        le.fit(combined_series)

        train_df[column] = le.transform(train_df[column])
        test_df[column] = le.transform(test_df[column])


    # At this point all of the categorical columns have been
    # completely filled and converted to integer values. There
    # are still numerical columns that need to be filled.
    #
    #              column        n_missing
    #         ----------------------------
    # train_df   LotFrontage      259
    #            MasVnrArea       8
    #            GarageYrBlt      81
    #
    # test_df    LotFrontage      227
    #            MasVnrArea       15
    #            BsmtFinSF1       1
    #            BsmtUnfSF        1
    #            TotalBsmtSF      1
    #            BsmtFullBath     2
    #            BsmtHalfBath     2
    #            GarageYrBlt      78
    #            GarageCars       1
    #            GarageArea       1


    # The missing values of MasVnrArea should be set to zero, 
    # since the MasVnrType values for these houses are 'None'.
    train_df.MasVnrArea.fillna(value=0, inplace=True)
    test_df.MasVnrArea.fillna(value=0, inplace=True)


    # The missing values of GarageYrBlt all belong to houses
    # that do not have garages. We will just set these to zero.
    train_df.GarageYrBlt.fillna(value=0, inplace=True)
    test_df.GarageYrBlt.fillna(value=0, inplace=True)


    # House Id 2121 and 2189 have no basement. Setting square 
    # foot values and bathroom numbers to zero.
    test_df.BsmtFinSF1.iloc[660] = 0
    test_df.BsmtFinSF2.iloc[660] = 0
    test_df.BsmtUnfSF.iloc[660] = 0
    test_df.TotalBsmtSF.iloc[660] = 0
    test_df.BsmtFullBath.iloc[660] = 0
    test_df.BsmtHalfBath.iloc[660] = 0

    test_df.BsmtFullBath.iloc[728] = 0
    test_df.BsmtHalfBath.iloc[728] = 0


    # For the house missing values for GarageCars and GarageArea,
    # the GarageType is 'Detchd'. Judging by the LotArea and
    # LotFrontage, it is probably a 1 car detached garage. The
    # mean area for a 1 car, detached garage in the test_df is
    # approximately 280sqft.
    test_df.GarageCars.iloc[1116] = 1
    test_df.GarageArea.iloc[1116] = 280.


    return train_df, test_df


def fill_missing_lotfrontages(train_df, test_df, random_state):
    """
        This function uses linear regression to predict the missing
        LotFrontage values for both the training and testing data sets. 
    """
    ################################################################
    # We need to fill in the missing LotFrontages for 259 houses in
    # the train data set, and 227 passengers in the test data set. 
    ################################################################

    train_df.LotFrontage.fillna(value=-1, inplace=True)
    test_df.LotFrontage.fillna(value=-1, inplace=True)


    # create a new training set from the original train/test sets
    train_frontage_train_df = train_df[train_df.LotFrontage != -1]
    train_frontage_train_df = train_frontage_train_df.drop(['SalePrice'], axis=1)
    train_frontage_test_df = test_df[test_df.LotFrontage != -1]

    new_train_df = pd.concat((train_frontage_train_df, train_frontage_test_df), axis=0)
    frontage_targets = new_train_df.LotFrontage
    new_train_df = new_train_df.drop(['LotFrontage'], axis=1)


    # create a new testing set from the original train/test sets
    test_frontage_train_df = train_df[train_df.LotFrontage == -1]
    test_frontage_train_df = test_frontage_train_df.drop(['SalePrice', 'LotFrontage'], axis=1)
    
    test_frontage_test_df = test_df[test_df.LotFrontage == -1]
    test_frontage_test_df = test_frontage_test_df.drop(['LotFrontage'], axis=1)


    # use xgb linear regression to predict missing values
    np.random.seed(random_state)
    x_train, x_test, y_train, y_test = train_test_split(new_train_df, frontage_targets)
    #################################################
    ### xgb regression parameters for grid search ###
    #################################################
    fixed_params = {'seed':random_state,
                    'objective':'reg:linear',
                    'learning_rate':0.1,
                    'gamma':0.01
                   }
                  
    lin_reg = XGBRegressor(**fixed_params)
    
    test_params = {'n_estimators':np.array([100, 250, 500]),
                   'max_depth':np.array([3, 5]),
                   'max_delta_step':np.array([0., 0.001]),
                   'reg_lambda':np.array([0., 0.01])
                  }

    
    grid_search = GridSearchCV(lin_reg, test_params, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(x_train, y_train)
    print 'Best Score: %0.3f'%grid_search.best_score_
    print 'Best Parameters:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])


    # make the predictions for training and test data sets
    frontage_predictions_train = grid_search.predict(test_frontage_train_df)
    frontage_predictions_test = grid_search.predict(test_frontage_test_df)
    

    # fill in the missing LotFrontage values in the main training set
    for (Id, frontage) in zip(test_frontage_train_df.Id, frontage_predictions_train):
        train_df.LotFrontage.iloc[Id - 1] = frontage


    # fill in the missing LotFrontage values in the main test set
    for (Id, frontage) in zip(test_frontage_test_df.Id, frontage_predictions_test):
        test_df.LotFrontage.iloc[Id - 1461] = frontage


    return train_df, test_df


def main(random_state, cv, train_df, test_df):
    """
        We will use the average between sklearn's LinearRegression and 
        xgboost's XGBRegressor predictions for the final price prediction.
    """
    train_df, test_df = fill_missing_lotfrontages(train_df, test_df, random_state)
    
    X = train_df.drop(['SalePrice'], axis=1)
    Y = train_df.SalePrice

    np.random.seed(random_state)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)

    #################################################
    ### lin regression parameters for grid search ###
    #################################################
    lin_fixed_params = {'n_jobs':-1
                       }

    lin_pca_params = {'random_state':random_state
                     }

    lin_pipeline = Pipeline([
            ('pca', PCA(**lin_pca_params)),
            ('lin', LinearRegression(**lin_fixed_params))
            ])

    lin_test_params = {'pca__n_components':np.array([55, 60, 70, 80]),
                       'pca__whiten':[True, False],
                       'lin__fit_intercept':[True, False], 
                       'lin__normalize':[True, False]
                      }
   
    np.random.seed(random_state)
    linreg_grid_search = GridSearchCV(lin_pipeline, lin_test_params, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error', cv=cv)
    np.random.seed(random_state)
    linreg_grid_search.fit(x_train, y_train)
    print 'Best Score: %0.5e'%linreg_grid_search.best_score_
    print 'Best Parameters:'
    best_parameters = linreg_grid_search.best_estimator_.get_params()
    for param_name in sorted(lin_test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])

    linreg_predictions_test = linreg_grid_search.predict(x_test)

    linreg_predictions = linreg_grid_search.predict(test_df)


    #################################################
    ### xgb regression parameters for grid search ###
    #################################################
    xgb_fixed_params = {'seed':random_state,
                        'objective':'reg:linear',
                        'learning_rate':0.01,
                        'reg_alpha':0.,
                        'max_depth':3,
                        'gamma':0.001,
                        'max_delta_step':0.
                       }

    xgbreg = XGBRegressor(**xgb_fixed_params)
    
    xgb_test_params = {'n_estimators':np.array([2500, 3500]), 
                       'reg_lambda':np.array([0.001, 0.01])
                      }
    
    xgbreg_grid_search = GridSearchCV(xgbreg, xgb_test_params, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error', cv=cv)
    xgbreg_grid_search.fit(x_train, y_train)
    print 'Best Score: %0.5e'%xgbreg_grid_search.best_score_
    print 'Best Parameters:'
    best_parameters = xgbreg_grid_search.best_estimator_.get_params()
    for param_name in sorted(xgb_test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])

    xgbreg_predictions_test = xgbreg_grid_search.predict(x_test)

    xgbreg_predictions = xgbreg_grid_search.predict(test_df)
    

    ##############################################################
    # For final predictions, take average of xgbreg and linreg
    ##############################################################
    predictions_test = (linreg_predictions_test + xgbreg_predictions_test) / 2.
    predictions = (linreg_predictions + xgbreg_predictions) / 2.
     
    
    ##########################################
    # Save predictions to csv file
    ##########################################
    prediction_df = pd.DataFrame(columns=['Id','SalePrice'], data=zip(test_df.Id, predictions))
    prediction_df.to_csv('submission_rs' + str(random_state).zfill(6) + '_cv' + str(cv) + '_combined_test1.csv', index=False)

    data = pd.DataFrame()
    data['true'] = y_test
    data['predicted'] = predictions_test
    
    sns.set_style(style='darkgrid')
    sns.jointplot('true', 'predicted', data=data, kind='reg', size=7)
    plt.plot([0,700000], [0,700000], linestyle='--')
    plt.xlim(xmin=0, xmax=700000)
    plt.ylim(ymin=0, ymax=700000)
#    plt.scatter(predictions_test, y_test, alpha=0.2)
#    plt.xlabel('Predicted Price (USD)', size=16)
#    plt.ylabel('True Price (USD)', size=16)
    plot_file = 'true_vs_predicted_rs' + str(random_state).zfill(6) + '_cv' + str(cv) + '_combined_test1.png'
    plt.savefig(plot_file)
    plt.clf()
    os.system('open ' + plot_file)

    print '*************************************'
    print '*************************************'
    print 'Random State: %s'%str(random_state).zfill(6)
    print 'RMSE        : %.5e'%np.sqrt(mean_squared_error(y_test, predictions_test))
    print '*************************************'
    print '*************************************'

if __name__ == '__main__':

    train_df, test_df = load_data()
    
    cv = 3
    n_iter = 1
    for ii in range(n_iter):
        
        random_state = 85065#np.random.randint(1, 100001)
        main(random_state, cv, train_df, test_df)
