import pandas as pd
import numpy as np
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
from sklearn import metrics

# ** Non testé **
# Nous pouvons organiser notre code en classes et méthodes (orienté objet) pour faciliter la réutilisation et la maintenance.
# 

class Trainer():

    def __init__(self, X, y, random_state=42):
        self.X = X
        self.y = y
        self.X_train = None
        self.random_state = random_state
        self.pipeline = None
 
    def _train(self):
        pass

    def train(self):
        if self.X_train is None:
            self.split()
        return self._train()
  
    def split(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=self.random_state)
        
 
class RegressionTrainer(Trainer):
    def _train(self, estimators, parameters, scoring='neg_mean_squared_error', cv=5):
        '''
        The function trains the model and returns the results
        A grid search is used to find the best parameters
        '''
        results = []
        
        for estimator in estimators.keys():  
            self.pipeline = Pipeline([('imputer', None), ('scaler', None), 
                                ('estimator', eval(estimator)())])
            
            parameters.update(estimators.get(estimator, {}))  
            grid_search = GridSearchCV(self.pipeline, parameters, cv=cv, scoring=scoring)
            grid_search.fit(self.X_train, self.y_train) 
            cv_results = grid_search.cv_results_
            best_estimator = grid_search.best_estimator_     
            results.append([estimator, grid_search.best_score_, grid_search.best_params_, cv_results, best_estimator])
        self.results = results
        return results
