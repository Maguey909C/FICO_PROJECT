#Author: Chase Renick
#Project: DS FICO SCORE PROJECT

import pandas as pd
import time
from warnings import simplefilter
import sys, getopt
import os
import wh_functions

def featureBlock(df, holdout_df):
    """
    INPUT: A dataFrame
    OUTPUT: Relevant features
    PURPOSE: Examines datafame taking only features with more than .49 of data present and then finds relevant features for model build
    """
    try:
        #Removes Features that have more than .49% of data missing
        df = wh_functions.removeEmptyFeatures(df)

        #Removes Features that have more than .49% of data missing on holdout set
        cleaned_holdout_df = wh_functions.removeEmptyFeatures(holdout_df)

        #Begins feature engineering process from training set. cleaned_holdout_df is ont used for feature engineering (ignore)
        relevant_features = wh_functions.featureEngineering(df, cleaned_holdout_df)
        relevant_features.append('y')

        return relevant_features, cleaned_holdout_df

    except:
        print ("Fatal Error within the preprocessingBlock, examine removing features or feature engineering")

def preprocessingBlock(df,target,relevant_features, holdout_df):

    try:
        #Splitting the Original Dataset into 80/20
        X_train, X_test, y_train, y_test, idx, idx2 = wh_functions.specificTTS(df,target,relevant_features) #Splitting

        #Performing Non Splitting on Holdout set because we want all the predicitons
        X_holdout, y_holdout = wh_functions.holdout_version(holdout_df, target, relevant_features)

        #Normalizing and Imputing based on original imputation strategy used in training set
        X_train, X_test, X_holdout = wh_functions.robustScaling(X_train, X_test, X_holdout) #Normalizing

        #Imputing Data for train, test, and holdout blocks
        Xtr, Xte, X_holdout = wh_functions.imputer(X_train, X_test, X_holdout) #Imputing

        return Xtr, y_train, Xte, y_test, X_holdout, y_holdout

    except:
        print ("Fatal Error with the modeling portion of scaling and imputing")

def main(argv):
    inputfile1 = ''
    inputfile2 = ''
    try:
        opts, args = getopt.getopt(argv,"hi:i:")
        inputfile1 = args[0]
        inputfile2 = args[1]
    except getopt.GetoptError:
        print ('Error in retreiving files for test.py -i <inputfile1> <inputfile2>')
        sys.exit(2)

    #Get two files and place in dataframes
    dirpath = os.getcwd()

    #Reading in the files from the current directory
    df = pd.read_csv(dirpath+"\\"+inputfile1)
    holdout_df = pd.read_csv(dirpath+"\\"+inputfile2)

    print ("\nPerforming Feature Engineering\n")
    # #Getting relevant features for model build
    relevant_features, cleaned_holdout_df = featureBlock(df, holdout_df)

    print ("Preprocessing\n")
    #Model Build
    Xtr, y_train, Xte, y_test, X_holdout, y_holdout = preprocessingBlock(df,'y',relevant_features, cleaned_holdout_df)

    print ("Building Model\n")
    # Generating Predictions for Test and Holdout Set
    test_predictions, holdout_predictions, regressor_test_accuracy, regressor_holdout_accuracy = wh_functions.baggedModel(Xtr, y_train, Xte, y_test, X_holdout, y_holdout)

    print ("Evaluating\n")
    #RSME & Accuracy Scores for Test Set
    wh_functions.rsme(y_test, test_predictions, "Test Set")
    print ("Test Set Regressor Accuracy: ", regressor_test_accuracy)

    print("\n")

    # RSME & Accuracy Scores for holdout Set
    wh_functions.rsme(y_holdout, holdout_predictions, "Holdout Set")
    print ("Holdout Set Regressor Accuracy: ", regressor_holdout_accuracy)

    print ("Generating Output")
    final_df = pd.DataFrame(holdout_predictions)
    final_df.columns = 'hs_predictions'
    #Saving the results to the current directory
    wh_functions.resultsFile(final_df)

if __name__ == '__main__':

    simplefilter(action='ignore', category=FutureWarning)
    start = time.time()
    main(sys.argv[1:])
    stop = time.time()
    print("Program took: ", stop-start, " seconds or, ",(stop-start)/60," minutes to complete.")
