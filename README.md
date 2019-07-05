#### Author: Chase R.
#### Date: 07/19

## DS FICO Project

### Introduction
The project is to demonstrate an approach to a credit scoring model based on an open source dataset that does not provide descriptions for the features beyond the label "y", which denotes FICO score. The purpose of the project is to demonstrate a data science approach towards building such a model, what questions would need to be asked, and what steps could be taken to score customers. 

### Approach
Since the features of this dataset were unknown, a number of assumptions were made about what they could represent such as 'x001' representing customer ID, some number of features were likely categorical, others were non categorical, among others. Many of these explanations and assumptions can be seen in the ipython notebook.

### Input Files
There is an ipython notebook file which walks through step by step an approach to building and scoring a model for this dataset based on the RSME value.  Additionally, two files were provided that essentially turn the ipython notebook code into more production ready code to be run from the command line.

To run the program from the command line follow the following steps:

1. Place the training data set, and the holdout data set in the same directory as the main.py and wh_funtions.py file.
2. From the anaconda prompt, cd into this directory and run the following command:
  
   python main.py training_set.csv holdout_set.csv
  
Note***

training_set.csv = whatever file you want the model to train on based on similar features

holdout_set.csv = the holdout file you want to test the model on 
   
### Model
Currently, the model takes about ~17 minutes to complete with the number of estimators set to 10. Although the RSME score would likely improve if we increased the number of estimators, the current time constraints for the project prevent further grid searching the parameters necessary to optimize the model.

### Output Files
When the model completes, it wil generate a csv file called, hold_out_set_results.csv, that contains the estimated FICO predictions from the holdout test set taken from the command line. Further joining of the customer ID number and other improvements to the model were not included by the author as it was not required for the project.

### Conclusion
As the evolution of the credit scoring model continues to evolve, it will be critical that data scientist and software developers do not build in biases into the predictive models when extending credit.  FICO has held competitions to help data scientists explain their "black box" models in part because the industry is heavily regulated and consumers will want to know why they are not being extended credit if denied.  https://community.fico.com/s/explainable-machine-learning-challenge.  There is much to do in this space in years to come, and it will be exciting to see what the outcomes will be.
   
 
