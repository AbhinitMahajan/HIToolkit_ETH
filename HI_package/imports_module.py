import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV as RandomSearchCV
from sklearn.feature_selection import RFE, SelectKBest, f_regression, RFECV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import Normalizer
