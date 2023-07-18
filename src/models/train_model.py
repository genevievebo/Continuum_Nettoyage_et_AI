import pandas as pd
import datetime
import logging
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, BaggingRegressor,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from data.dataset import OatsDataset


def train_regression(X_train, y_train, estimators, parameters, scoring='r2_score', cv=5):
    '''
    The function trains the model and returns the results
    A grid search is used to find the best parameters
    '''
    results = []
    
    for estimator in estimators.keys():
        params = parameters.copy()
        pipeline = Pipeline([('imputer', None), ('scaler', None), 
                            ('estimator', eval(estimator)())])
        
        params.update(estimators.get(estimator, {}))  
        grid_search = GridSearchCV(pipeline, parameters, cv=cv, scoring=scoring)
        grid_search.fit(X_train, y_train) 
        cv_results = grid_search.cv_results_
        best_estimator = grid_search.best_estimator_     
        results.append([estimator, grid_search.best_score_, grid_search.best_params_, cv_results, best_estimator])
    
    return results


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    

    transformers = {
            'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
            'imputer': [SimpleImputer(), IterativeImputer()]
    }

    estimators = {
        'RandomForestRegressor': {'estimator__max_depth': [4, 6, 8, 10, 12]},
        'LinearRegression': {},
        'DecisionTreeRegressor': {},
        'KNeighborsRegressor': {},
        'GradientBoostingRegressor': {},
        'SVR': {}
    }
    
    
    dataset = OatsDataset()
    dataset.load()
   
    logger.info(f'Loading {dataset.name} - {dataset.path}')

    results = train_regression(dataset.X, dataset.y, estimators, transformers)
    results = pd.DataFrame(results, columns=['estimator', 'best_score', 'best_params', 'cv_results', 'best_estimator'])
    today = datetime.datetime.now().strftime("%Y%m%d")
    dest = F'models/{today}'
    if not os.path.exists(dest):
        os.makedirs(dest)
    results.to_csv(F'{dest}/results.{today}.csv')
    
    logger.info(f'Training done  - Results saved in {dest}/results.{today}.csv ')
    
    
    
