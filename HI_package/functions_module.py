# Importing everything again
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize, Normalizer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE, SelectKBest, f_regression, RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Functions to remove outliers
def remove_outliers_iqr(data, factor=1.5):
    if isinstance(data, pd.DataFrame):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (data < (Q1 - factor * IQR)) | (data > (Q3 + factor * IQR))
        return data[~is_outlier.any(axis=1)]
    elif isinstance(data, np.ndarray):
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        is_outlier = (data < (Q1 - factor * IQR)) | (data > (Q3 + factor * IQR))
        return data[~is_outlier.any(axis=1)]
    else:
        raise TypeError("Expected numpy.ndarray or pandas.DataFrame")

def remove_outliers_zscore(data, threshold=3.0):
    """
    Remove outliers using Z-score.
    """
    z_scores = np.abs(zscore(data))
    return data[(z_scores < threshold).all(axis=1)]

def remove_outliers_isolation_forest(data, contamination=0.05):
    """
    Remove outliers using Isolation Forest.
    """
    iso_forest = IsolationForest(contamination=contamination)
    is_inlier = iso_forest.fit_predict(data)
    return data[is_inlier == 1]

# Preprocessing functions
def preprocess_min_max_scaling(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def preprocess_standard_scaling(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def preprocess_normalization(X):
    return normalize(X)

def preprocess_l2_normalization(X):
    normalizer = Normalizer(norm='l2')
    return normalizer.fit_transform(X)

# Feature selection functions
def feature_selection_rfe(estimator, X_train, y_train, num_features):
    rfe = RFE(estimator=estimator, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)
    return rfe.transform(X_train), rfe.support_

def feature_selection_selectkbest(X_train, y_train, num_features):
    skb = SelectKBest(f_regression, k=num_features)
    return skb.fit_transform(X_train, y_train)

def feature_selection_rfe_cv(estimator, X_train, y_train):
    rfecv = RFECV(estimator=estimator, scoring='r2')
    X_train_selected = rfecv.fit_transform(X_train, y_train)
    return X_train_selected, rfecv.support_

def get_selected_feature_names(X, support_mask):
    return list(X.columns[support_mask])


# Model specific tuning functions
kfold = KFold(n_splits=4, shuffle=True, random_state=42)

def tune_parameters_pls(X_train_selected, y_train):
    model = PLSRegression()
    n_features = X_train_selected.shape[1]
    param_grid = {
        'n_components': range(1, 30),  # Adjusted the range here
        'scale': [False],
        #'scale': [True, False],
    }
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)

    return best_model

def tune_parameters_svm(X_train_selected, y_train):
    model = SVR()
    param_grid = {
    'C': [0.001, 0.01,0.1, 1,100,200],
    'kernel': ['linear'],
    }
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error',cv=5)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model

def tune_parameters_xgboost(X_train_selected, y_train):
    model = xgb.XGBRegressor()
    param_grid = {
        'n_estimators': [ 100, 300, 500, 700, 900],
        'max_depth': [4, 5, 7 , 9],
        'learning_rate': [0.001, 0.01, 0.1, 0.05],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model

def tune_parameters_randomforest(X_train_selected, y_train):
    model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 300, 500,1000],
        'max_depth': [4, 6, 8, None],  # None for no limit
        'min_samples_split': [4, 6, 8],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model

# def tune_parameters_lasso(X_train_selected, y_train):
#     model = Lasso()
#     param_grid = {
#         'alpha': [0.001,0.01, 0.1, 1,100],
#         'max_iter' : [1000000]
#     }
#     grid_search = GridSearchCV(model, param_grid, scoring='r2')
#     grid_search.fit(X_train_selected, y_train)
#     best_model = grid_search.best_estimator_
#     return best_model

def tune_parameters_lasso(X_train_selected, y_train):
    # Param Grid
    model = Lasso()
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [10000000]
    }
    grid_search = GridSearchCV(model, param_grid, scoring='r2')
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    feature_names = list(df_turb.iloc[:,1:].columns)
    # Identify and Visualize Important Features
    coef = best_model.coef_
    important_indices = np.where(coef != 0)[0]
    unimportant_indices = np.where(coef == 0)[0]

    # Extract feature names using indices
    important_features = [feature_names[i] for i in important_indices]
    unimportant_features = [feature_names[i] for i in unimportant_indices]

    # Print the important features
    print("Important features:", important_features)
    print("Number of unimportant features:", len(unimportant_features))

    # Plot the feature importances
    plt.figure(figsize=(20, 8))
    feature_importance = pd.Series(coef, index=feature_names).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importance')
    plt.show()

    return best_model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
