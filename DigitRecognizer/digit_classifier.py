import numpy as np
import pandas as pd
import os

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def load_data():
    
	train_df = pd.read_csv('./InputFiles/train.csv')
	test_df = pd.read_csv('./InputFiles/test.csv')

	return train_df, test_df


def main(train_df, test_df, scoring, cv, random_state):

    # split the data for cross validation
    X = train_df.drop(['label'], axis=1)
    Y = train_df.label
        
    np.random.seed(random_state) # so train_test_split is reproducible
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    
    # scale the data to zero mean and unit variance
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #################################################
    ### xgb classifier parameters for grid search ###
    #################################################
    svd_fixed_params = {'random_state':random_state,
                        'n_components':100
					   }
	
    clf_fixed_params = {'random_state':random_state,
		               }

    svd = TruncatedSVD(**svd_fixed_params)
    clf = MLPClassifier(**clf_fixed_params)

    pipeline = Pipeline([('svd', svd),
	    				 ('clf', clf)
		    			 ])

    test_params = {'svd__n_iter':np.array([5, 10, 15, 25]),
                   'clf__shuffle':[True],
                   'clf__learning_rate':['constant'],
                   'clf__alpha':np.logspace(-1, -1, 1)
			      }

    grid_search = GridSearchCV(pipeline, test_params, n_jobs=-1, verbose=1, scoring=scoring, cv=cv)
    grid_search.fit(x_train_scaled, y_train)

    # print out grid search best score and parameters
    print 'Best %s score: %0.3f'%(scoring, grid_search.best_score_)
    print 'Best Parameters:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])

    # check how classifier performs of the rest of the training set
    predictions_test = grid_search.predict(x_test_scaled)
    print "*************"
    print "* RS-%s *"%str(random_state).zfill(6)
    print "* accuracy  *"
    print "*           *"
    print "*  %.5f  *"%accuracy_score(y_test, predictions_test)
    print "*           *"
    print "*************"


    test_df_scaled = scaler.transform(test_df)
    predictions = grid_search.predict(test_df_scaled)
    ImageIds = np.arange(1, len(predictions) + 1)
    prediction_df = pd.DataFrame(columns=['ImageId', 'Label'], data=zip(ImageIds, predictions))
    prediction_df.to_csv('./SubmissionFiles/digits_submission_rs' + str(random_state).zfill(6) + '_cv' + str(cv) + '.csv', index=False)



if __name__ == '__main__':

    n_iter = 1
    scoring = 'accuracy' # evaluation metric for grid search
    cv = 3 # number of cross validation sets

    train_df, test_df = load_data()

    for ii in range(n_iter):
        random_state = 19790#np.random.randint(1, 100001)
        main(train_df, test_df, scoring, cv, random_state)


