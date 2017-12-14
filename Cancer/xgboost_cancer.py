import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def load_data():
    """

    """
    
    train_df = pd.read_csv('training_variants')
    train_text = pd.read_csv('training_text', sep='\|\|', engine='python', names=['textID', 'Text'], skiprows=1)
    test_df = pd.read_csv('test_variants')
    test_text = pd.read_csv('test_text', sep='\|\|', engine='python', names=['textID', 'Text'], skiprows=1)


    # Add the text features to the training/test data frames
    train_df = pd.concat((train_df, train_text), axis=1)
    train_df = train_df.drop(['textID'], axis=1)
    test_df = pd.concat((test_df, test_text), axis=1)
    test_df = test_df.drop(['textID'], axis=1)

    # Build a data frame of all of the unique texts
    corpora = pd.concat((train_text, test_text), axis=0)
    unique_corpora = pd.Series(corpora.Text.unique())

    # Create a feature that contains the relevant literature
    # from the unique corpora that pertains to each train_df
    # instance's Gene & Variation variable.
    # FIXME this needs some work and might no be doable for now.
    """
    train_df['RelevantTexts'] = 0
    
    for ii in xrange(train_df.shape[0]):
        gene = train_df.Gene.iloc[ii]
        variation = train_df.Variation.iloc[ii]
        print gene, variation
        thing = [jj for (jj, corpus) in unique_corpora.iteritems() if (corpus.__contains__(gene) or corpus.__contains__(variation))]
        print len(thing)

    exit()
    """

    return train_df, test_df, unique_corpora

def vectorize_text(train_df, test_df, unique_corpora):
    """

    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    print 'Vectorizing text...'
    all_vectorized = vectorizer.fit_transform(unique_corpora).todense()
    vectorized_train_text = vectorizer.transform(train_df.Text).todense()
    vectorized_test_text = vectorizer.transform(test_df.Text).todense()
    print 'Finished vectorizing!\n'

    return vectorized_train_text, vectorized_test_text, all_vectorized

def pca_text(train_df, test_df, vectorized_train_text, vectorized_test_text, all_vectorized, n_components, random_state):
    """

    """

    print 'Reducing vectorized text to %d dimension(s)...'%n_components
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(all_vectorized)
    reduced_train_text = pca.transform(vectorized_train_text)
    reduced_test_text = pca.transform(vectorized_test_text)
    print 'Finished dimension reduction!\n'

    for ii in xrange(n_components):
        new_column = 'pca' + str(ii + 1)

        train_df[new_column] = reduced_train_text[:,ii]
        test_df[new_column] = reduced_test_text[:,ii]

    # We don't need the actual text anymore
    train_df = train_df.drop(['Text'], axis=1)
    test_df = test_df.drop(['Text'], axis=1)

    return train_df, test_df
    

def main(train_df, test_df, test_IDs, cv, pca_comp, random_state):

    # Encode all of the categorical data into numerical data
    categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        le = LabelEncoder()
        combined_series = pd.concat((train_df[column], test_df[column]), axis=0)
        le.fit(combined_series)

        train_df[column] = le.transform(train_df[column])
        test_df[column] = le.transform(test_df[column])



    X = train_df.drop(['Class'], axis=1)
    Y = train_df.Class

    np.random.seed(random_state)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)

    #####################################################
    ### xgb classification parameters for grid search ###
    #####################################################
    xgb_fixed_params = {'seed':random_state,
                        'objective':'multi:softprob',
                        'reg_alpha':0.,
                        'max_delta_step':0.
                       }

    test_params = {'n_estimators':np.array([50, 75, 100]), 
                   'max_depth':np.array([3, 5, 7]),
                   'gamma':np.array([0.001, 0.01]),
                   'reg_lambda':np.array([0., 0.001, 0.01]),
                   'learning_rate':np.array([0.001, 0.01, 0.1])
                  }
    
    xgb = XGBClassifier(**xgb_fixed_params)

    grid_search = GridSearchCV(xgb, test_params, n_jobs=-1, verbose=1, scoring='neg_log_loss', cv=cv)
    grid_search.fit(x_train, y_train)
    print 'Best Score: %0.5e'%grid_search.best_score_
    print 'Best Parameters:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(test_params.keys()):
        print '\t%s: %r'%(param_name, best_parameters[param_name])

    predictions_test = grid_search.predict_proba(x_test)
    print "Predictions"
    print "============"
    print predictions_test

    predictions = grid_search.predict_proba(test_df)
#   predictions = pd.DataFrame(predictions, columns=['class1','class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'])

#    final_df = pd.concat((test_IDs, predictions), axis=1)
    prediction_df = pd.DataFrame(columns=['class1','class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'], data=predictions)
    prediction_df.to_csv('test_submission_rs' + str(random_state).zfill(6) + '_cv' + str(cv) + '_pcaComp' + str(pca_comp) + '_xgboost.csv', index=True, index_label='ID')

#    data = pd.DataFrame()
#    data['true'] = y_test
#    data['predicted'] = predictions_test
#    
#    sns.set_style(style='darkgrid')
#    sns.jointplot('true', 'predicted', data=data, kind='reg', size=7)
#    plt.plot([0,700000], [0,700000], linestyle='--')
#    plt.xlim(xmin=0, xmax=700000)
#    plt.ylim(ymin=0, ymax=700000)
##    plt.scatter(predictions_test, y_test, alpha=0.2)
##    plt.xlabel('Predicted Price (USD)', size=16)
##    plt.ylabel('True Price (USD)', size=16)
#    plot_file = 'true_vs_predicted_rs' + str(random_state).zfill(6) + '_cv' + str(cv) + '_xgboost.png'
#    plt.savefig(plot_file)
#    plt.clf()
#    os.system('open ' + plot_file)

    print '*************************************'
    print '*************************************'
    print 'Random State: %s'%str(random_state).zfill(6)
#    print 'Neg Log Loss: %.5e'%mean_squared_error(y_test, predictions_test)
    print 'GridSearch  : %.5f (log loss)'%grid_search.best_score_
    print '*************************************'
    print '*************************************'

if __name__ == '__main__':

    train_df, test_df, unique_corpora = load_data()
    # Drop ID columns
    train_df = train_df.drop(['ID'], axis=1)
    test_IDs = test_df.ID
    test_df = test_df.drop(['ID'], axis=1)

#    print train_df.count()
#    print test_df.count()
#    print 'Train\n',train_df
#    print 'Test\n',test_df

    vectorized_train_text, vectorized_test_text, all_vectorized = vectorize_text(train_df, test_df, unique_corpora)

    cv = 3
    n_iter = 1
    pca_components = [10]#[10, 20, 30, 50]
    for ii in range(n_iter):
        
        random_state = 12433#np.random.randint(1, 100001)
#        vectorized_train_text, vectorized_test_text, all_vectorized = vectorize_text(train_df, test_df, unique_corpora)
        # Iterate over several values of PCA dimensions to see what scores best
        for pca_comp in pca_components:

            pca_train_df, pca_test_df = pca_text(train_df, test_df, vectorized_train_text, vectorized_test_text, all_vectorized, pca_comp, random_state)
            
            main(pca_train_df, pca_test_df, test_IDs, cv, pca_comp, random_state)


