#### Author: Chase R.
#### Date: 07/19

## DS FICO Project

### Introduction
The project is to demonstrate an approach to a credit scoring model based on an open source dataset that does not provide descriptions for the features beyond the label "y", which denotes FICO score. The purpose of the project is to demonstrate a data science approach towards building such a model, what questions would need to be asked, and what steps could be taken to score customers. 

### Approach
Since the features of this dataset were unknown, a number of assumptions were made about what they could represent such as 'x001' representing customer ID, some number of features were likely categorical, others were non categorical, among others. Many of these explanations and assumptions can be seen in the ipython notebook.

### Input Files
There is an ipython notebook file which walks through step by step an approach to building and scoring a model for this dataset based on the RSME value.
The two python files relate to the ipython notebook code which was transformed into more production level code to be run from the command line. In order run the program from the command line, you must have 2 csv files to compare within the same directory as the main.py file and the wh_functions.py file. 

1. Place the training data set, and the holdout data set in the same directory as the main.py and wh_funtions.py file.
2. From the anaconda prompt, cd into this directory and run the following command:
  
   python main.py input_file_1.csv input_file_2.csv
  
Note***

input_file_1.csv = whatever file you want the model to train on based on similar features

input_file_2.csv = the holdout file you want to test the model on 
   
### Model
Based on my current hardware of Intel i7 @2.90GHz and 32GB RAM, model specs oRandom Forest Regressor estimators=100 and the boosted estimators=10, the model takes about ~5 minutes to complete without any multiprocessing or parallel processing implementations.

The model could improve if we increaed the number of estimators, but in the interest of time and the marginal benefits of such improvement in performance, I chose to to favor run time over the ideal performance. The hyperparameters were grid searched, but again in the interest of time, there are more combinations that could to attempted with more compute power.

### Output

As the model runs it will indicate in the console what step it is currently in during the build process. As the model reaches its completion it will print to the console the RSME and Regessor Accuracy to the console for input_file_1.csv and then the same evaluation metrics for input_file_2.csv .  Please note thta input_file_2.csv does not undergo the test train or split process since we want to predict on this dataset all FICO values related to customers.

When the model completes, it wil generate a pandas dataframe from the predictions generated from the input_file_2.csv, and it will save the predictoins in a file called, hold_out_set_results.csv.  hold_out_set_results.csv contains both the estimated FICO predictions as well as a rounded version of them, in case the user prefers the whole number to compare FICO. Joining other attributes to the results such as customer ID number and other improvements to the model were beyond the scope of this project.

### Conclusion
The model performed best with a gridsearched Random Forest Regressor ensemble technique using AdaBoost.  The RSME score was around 28.5 and the Regressor Accuracy near 94.3%. The model and continue to improve with more grid searching and further knowledge of which features were categorical or not, but for the moment it performs sufficiently well for this project.

As the evolution of the credit scoring model continues to evolve, it will be critical that data scientist and software developers do not build in biases into the predictive models when extending credit.  FICO has held competitions to help data scientists explain their "black box" models in part because the industry is heavily regulated and consumers will want to know why they are not being extended credit if denied.  https://community.fico.com/s/explainable-machine-learning-challenge.  There is much to do in this space in years to come, and it will be exciting to see what the outcomes will be.
   
 
