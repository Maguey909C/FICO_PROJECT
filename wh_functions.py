import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from warnings import simplefilter


def removeEmptyFeatures(df):
    """
    INPUT: A dataframe
    OUTPUT:
    PURPOSE:
    """

    nan_df = df.iloc[:,1:-1].isna().sum(axis=0).reset_index().rename(columns={'index':'features', 0:'nan_count'})
    nan_df['nan_percentage'] = (nan_df['nan_count'] / 100000)
    remove_features = list(nan_df[nan_df['nan_percentage']>.49].features)
    df2 = df.drop(remove_features,axis=1)

    return df2

def tts(df, target):
    """
    INPUT: A dataframe, and your target variable
    OUTPUT: Train test split of dataframe based on
    """

    X = df.loc[:, df.columns != str(target)].iloc[:,1:]
    y = df.loc[:, df.columns == str(target)].values.flatten()
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test, idx1, idx2

def robustScaling(X_train,X_test):
    """
    INPUT:Training set from X_train, Training set X_test
    OUTPUT: Normalized Scaled versions of the X_train and X_test
    PURPOSE: To normalize / scale data specifically in regards to potential outliers
    """
    std = RobustScaler()
    std.fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test) #Applying the same scaling method to the test set as training set

    return X_train, X_test

def imputer(X_train, X_test):
    """
    INPUT:Training set from X_train, Training set X_test
    OUTPUT:Imputing values for the NaNs based on the median of what is contained in the
    PURPOSE: To estimate values for NaNs which otherwise would have to be removed from the dataset
    """
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    return X_train, X_test

def randomForestFeatures(df, X_train, y_train):
    """
    INPUT: A dataframe to, X_train, y_train
    OUTPUT: A list of tuples ranking the feature importance generated from the
    PURPOSE:
    """
    names = df.iloc[:,1:-1].columns

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    tups = (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))
    return tups

def getThresholdDictonary(features, threshold):
    """
    INPUT: The list of tuples generated from the Random Forest Regressor feature extraction,
    a threshold for what features you want to include
    OUTPUT: A dictionary of the features and their importances, as well as a list of them unordered
    PURPOSE: To allow for more feature exploration and getting a list of features to filter on when building model
    """
    relevant_features = []
    for i in range(len(features)):
        v,k = (features[i])

        if v >= threshold:
            relevant_features.append(k)

    return relevant_features

def featureEngineering(df):
    """
    INPUT: A dataframe
    OUTPUT: A list of relevant features based on Random Forest Regressor
    PURPOSE: To determine what are the most relevant features related to the dataset to drive the model
    """
    #Splitting Data into 80/20
    X_train, X_test, y_train, y_test, idx1, idx2 = tts(df,'y')

    #Scaling / Normalizing the data to minimize impact of outliers
    X_train, X_test = robustScaling(X_train, X_test)

    #Imputing Values on the training set and applying same std and mean to test set
    Xtr, Xte = imputer(X_train, X_test)

    #Feature Selection with Random Forest Regressor
    impt_features = randomForestFeatures(df, Xtr, y_train)

    #Selecting only those features with an importance > .005
    relevant_features = getThresholdDictonary(impt_features, .001)

    return relevant_features

def specificTTS(df, target, relevant_features):
    """
    INPUT:A dataframe
    OUTPUT:
    """
    X = df[relevant_features[:-1]].values
    y = df.loc[:, df.columns == str(target)].values.flatten()
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test, idx1, idx2

def baggedModel(X_train, y_train, new_dataset):
    """
    INPUT:
    OUTPUT:
    """
    rf_reg = RandomForestRegressor()
    bagged_rf_rg = BaggingRegressor(base_estimator=rf_reg,
                                             n_estimators=100,
                                             random_state=123)
    bagged_rf_rg.fit(X_train, y_train)

    return bagged_rf_rg.predict(new_dataset)

def rsme(actuals, predictions):
    """
    INPUT: Predictions to the model
    OUTPUT: Root mean squared error
    PURPOSE: To measure the performance of the model
    """

    return np.round(np.sqrt(mean_squared_error(actuals, predictions)),2)

def resultsFile(df):
    """
    INPUT:A dataframe
    OUTPUT: A csv file of that dataframe placed in the current directory where this file sits
    PURPOSE: Generate results to a dataframe based on output
    """

    return df.to_csv("holdout_set_results.csv")
