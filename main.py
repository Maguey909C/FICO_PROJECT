import pandas as pd
import time
from warnings import simplefilter
import sys, getopt
import os
import wh_functions

def featureBlock(df):
    """
    INPUT: A dataFrame
    OUTPUT: Relevant features
    PURPOSE: Examines datafame taking only features with more than .49 of data present and then finds relevant features for model build
    """
    try:
        df = wh_functions.removeEmptyFeatures(df)
        relevant_features = wh_functions.featureEngineering(df)
        relevant_features.append('y')
        return relevant_features

    except:
        print ("Fatal Error within the preprocessingBlock, examine removing features or feature engineering")

def preprocessingBlock(df,label,relevant_features):

    try:
        X_train, X_test, y_train, y_test, idx, idx2 = wh_functions.specificTTS(df,label,relevant_features) #Splitting
        X_train, X_test = wh_functions.robustScaling(X_train, X_test) #Normalizing
        Xtr, Xte = wh_functions.imputer(X_train, X_test) #Imputing

        return Xtr, y_train, Xte, y_test

    except:
        print ("Fatal Error with the modeling portion of scaling and imputing")

def modelingBlock(X_train, y_train, y_test, holdout_df):

    try:
        #Generating Predictions
        predictions = wh_functions.baggedModel(X_train, y_train, holdout_df)

        #Generating the RSME Score
        rsme_score = wh_functions.rsme(y_test, predictions)

        return rsme_score

    except:
        print ("Fatal Error in modeling portion")

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
    print ("Path to second file", dirpath+"\\"+inputfile2)
    # df2 = pd.read_csv(dirpath+"\\"+inputfile2)

    print ("Performing Feature Engineering")
    #Getting relevant features for model build
    relevant_features = featureBlock(df)

    print ("Building Real Model")
    #Model Build
    Xtr, y_train, Xte, y_test = preprocessingBlock(df,'y',relevant_features)

    print ("Generating Predictions")
    #Generating Predictions
    rsme_score = modelingBlock(Xtr, y_train, y_test, Xte)

    print ("RSME Score: ", rsme_score)

    #Saving the results to the current directory
    # wh_functions.resultsFile(df)

if __name__ == '__main__':

    simplefilter(action='ignore', category=FutureWarning)
    start = time.time()
    main(sys.argv[1:])
    stop = time.time()
    print("Program took: ", stop-start, " seconds or, ",(stop-start)/60," minutes to complete.")
