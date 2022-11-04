# Credit Risk Calculator:
* I created a tool that models and estimates credit risk based on credit data provided by Data Camp.
* The dataset contained over 32,000 observations with 11 features.
* I then tested both logistic regression and gradient boosted tree models to reach the produce best metrics. 
* Finally I built a client facing API using Flask.

## Code Information / Resources Used:
**Python Version:** 3.9.7
**Packages Used:** pandas, matplotlib, numpy, sklearn, xgboost, pickle, flask, json,  
**Flask Productionization Article:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Features from the data:
*	Person Age
* Person Income
* Person Home Ownership
* Person Employment Length
* Loan Intent
* Loan Grade
* Loan Amount
* Loan Interest Rate
* Loan Percent Income
* Person Default On File
* Person Credit History Length

## Data Cleaning / Feature Engineering
Once obtained, I cleaned up the data so that it was usable for EDA and modelling. I made the following changes:

*	Filled in missing data for 'Employment Length' using the median of the values.
*	Dropped any observation where 'interest rate' was null.
*	Removed any outliers for the columns of 'Employment Length' and 'Person Age'.
*	One Hot Encoded all non-numeric columns including:
    * Person Home Ownership
    * Loan Intent
    * Loan Grade
    * Person Default On File
* Concatenated numerical columns along with the OHE columns created to create a dataframe fit for modelling.

## Model Building 

I began by splitting the data into train and tests sets with a test size of 0.4.   

I tried both a logistic regression model and a gradient boosted tree model where I compared recall scores. I chose this as I wanted my model to seek out all relevant cases within the dataset. For the application of detecting loan defaults, there is a large financial penalty for predicting no default when it actually does.
 
## Model performance
The gradient boosted tree model performed best on the test set. 
*	**Logistic Regression** : Recall = 0.94
*	**Gradient Boosted Trees**: Recall = 0.98

## Fine Tuning

With the gradient boosted tree model, I then continud to fine-tune the model:
* I removed features that had a low importance weigting to reduce dimensionality of the model. This was to stop the model overfitting to the training set.
* I undersampled the data. There was a large difference of classification value counts (16829:850) within the training data. This would remove the bias of the data.
* I found the optimum prediction decision threshold. To achieve this I used the model to predict probability of default instead a discrete value. I then looped through a list of threshold values and calculated expected profits from these predictions.
  * **Optimum Threshold Value** : 0.80

## Productionization 
I built a Flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from representing the model features, and outputs a 1 or 0 depending on whether the loan is expected to default or not. 
