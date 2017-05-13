import numpy as np
import pandas as pd
import time

from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    train_df = pd.read_csv('train.csv')
#    test_df = pd.read_csv('test.csv')

    # make separate data frame for digits and take out of training set
    X = train_df.drop(['label'], axis=1)
    y = train_df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    pipeline = Pipeline([
                        ('ss', StandardScaler()),
                        ('mlp', MultilayerPerceptronClassifier(n_hidden[150, 100], alpha=0.1))
                        ])

    print cross_val_score(pipeline, X, y, n_jobs=-1)

    #train_data = train_df.values.astype(np.uint8)
    #target_data = target_df.values.astype(np.uint8)
    #test_data = test_df.values.astype(np.uint8)

#    svc = SVC()
#    print 'Fitting SVC...'
#    t0 = time.time()
#    svc.fit(X_train, y_train)
#    print 'time elapsed: %.1f\n'%(time.time() - t0)
#
#    print 'Starting SVC...'
#    output = svc.predict(test_data)
#    print 'Finished with predictions.'
#
#    ImageIds = np.arange(1, 28001)
#    S = Series(output, index=ImageIds, dtype=np.uint8)
#    S.to_csv('kNeighbors_results.csv', header=True, index_label=['ImageId', 'Label'])
