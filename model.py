# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:06:44 2022

@author: Dan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('cr_loan_clean.csv')

# OHE non-numeric columns
df_num = df.select_dtypes(exclude=['object'])
df_str = df.select_dtypes(include=['object'])
df_str_onehot = pd.get_dummies(df_str)
df = pd.concat([df_num, df_str_onehot], axis=1)

# Create the X and y data sets
X = df.drop('loan_status', axis = 1)
y = df[['loan_status']]

# Test train split
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Log reg predictions and metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score
y_pred_lr = lr.predict_proba(X_test)
pred_df_lr = pd.DataFrame(y_pred_lr[:,1], columns = ['y_pred'])
pred_df_lr['loan_status'] = pred_df_lr['y_pred'].apply(lambda x: 1 if x > 0.5 else 0)
print(pred_df_lr['loan_status'].value_counts())
print(classification_report(y_test, pred_df_lr['loan_status'], target_names=['Non-Default', 'Default']))
print(precision_recall_fscore_support(y_test,pred_df_lr['loan_status']))
print(precision_recall_fscore_support(y_test,pred_df_lr['loan_status'])[:2])
print(lr.score(X_test, y_test))

# ROC curve
fallout, sensitivity, thresholds = roc_curve(y_test, y_pred_lr[:,1])
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()
auc = roc_auc_score(y_test, y_pred_lr[:,1])

# XGBoost
import xgboost as xgb
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train, np.ravel(y_train))

# XGBoost predictions and metrics
y_pred_gbt = gbt.predict(X_test)
print(classification_report(y_test, y_pred_gbt, target_names=['Non-Default', 'Default']))
print(precision_recall_fscore_support(y_test,y_pred_gbt))
print(precision_recall_fscore_support(y_test,y_pred_gbt)[:2])
print(gbt.score(X_test, y_test))
print(roc_auc_score(y_test, y_pred_gbt))


# Feature importance
print(gbt.get_booster().get_score(importance_type = 'weight'))
xgb.plot_importance(gbt, importance_type = 'weight')
plt.show()

# Cross validation
DTrain = xgb.DMatrix(X_train, label = y_train)
params = {'objective': 'binary:logistic', 'seed': 123, 'eval_metric': 'auc'}
cv = xgb.cv(params, DTrain, num_boost_round = 600, nfold=10, shuffle = True)
print(cv)
print(cv.head())
print(np.mean(cv['test-auc-mean']))
plt.plot(cv['test-auc-mean'])
plt.title('Test AUC Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test AUC Score')
plt.show()

cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)
print(cv_scores)
print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# Undersampling training data
X_y_train = pd.concat([X_train.reset_index(drop = True), y_train.reset_index(drop = True)], axis = 1)
count_nondefault, count_default = X_y_train['loan_status'].value_counts()
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]
nondefaults_under = nondefaults.sample(count_default)
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True), defaults.reset_index(drop = True)], axis = 0)
print(X_y_train_under['loan_status'].value_counts())

# Testing undersampled performance
X_U = X_y_train_under.drop('loan_status', axis = 1)
y_U = X_y_train_under[['loan_status']]
X_train_U, X_test_U, y_train_U, y_test_U = train_test_split(X_U, y_U, test_size=.4, random_state=123)
gbt_undersampled = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train_U, np.ravel(y_train_U))
y_pred_gbt_U = gbt.predict(X_test_U)
print(classification_report(y_test_U, y_pred_gbt_U, target_names=['Non-Default', 'Default']))
print(roc_auc_score(y_test_U, y_pred_gbt_U))


# Comparing models
print(classification_report(y_test, pred_df_lr['loan_status'], target_names=['Non-Default', 'Default']))
print(classification_report(y_test, y_pred_gbt, target_names=['Non-Default', 'Default']))

print(precision_recall_fscore_support(y_test,pred_df_lr['loan_status'], average = 'macro')[2])
print(precision_recall_fscore_support(y_test,y_pred_gbt, average = 'macro')[2])

fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, pred_df_lr['loan_status'])
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, y_pred_gbt)
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

# Calibration curve
from sklearn.calibration import calibration_curve
x_lr, y_lr = calibration_curve(y_test, pred_df_lr['loan_status'], n_bins = 10)
x_gbt, y_gbt = calibration_curve(y_test, y_pred_gbt, n_bins = 10)

plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(x_lr, y_lr,
         's-', label='%s' % 'Logistic Regression')
plt.plot(x_gbt, y_gbt,
         's-', label='%s' % 'Gradient Boosted Tree')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()

# Gradient boosted tree is by far the best model
# Will use this model along with feature thats have an F score of over 300
X = df[['person_age','person_emp_length','loan_amnt','loan_percent_income','loan_int_rate','person_income']]
y = df[['loan_status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Acceptance rates
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train, np.ravel(y_train))
y_pred = gbt.predict_proba(X_test)
pred_df = pd.DataFrame(y_pred[:,1], columns = ['y_pred'])
pred_df['loan_status'] = pred_df['y_pred'].apply(lambda x: 1 if x > 0.5 else 0)

threshold_85 = np.quantile(pred_df['y_pred'], 0.85)
pred_df['pred_loan_status'] = pred_df['y_pred'].apply(lambda x: 1 if x > threshold_85 else 0)
print(pred_df['pred_loan_status'].value_counts())

plt.hist(pred_df['y_pred'], color = 'blue', bins = 40)
threshold = np.quantile(pred_df['y_pred'], 0.85)
plt.axvline(x = threshold, color = 'red')
plt.show()

# Calculating the bad rate
accepted_loans = pred_df[pred_df['pred_loan_status'] == 0]
print(np.sum(accepted_loans['loan_status']) / accepted_loans['loan_status'].count())

# Implementing a strategy table to find optimum threshold
accept_rates = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]
thresholds = []
bad_rates = []
num_accepted_loans = []
num_default_loans = []
num_non_default_loans = []
est_val = []
for rate in accept_rates:
    thresh = np.quantile(pred_df['y_pred'], rate).round(3)
    thresholds.append(np.quantile(pred_df['y_pred'], rate).round(3))
    pred_df['pred_loan_status'] = pred_df['y_pred'].apply(lambda x: 1 if x > thresh else 0)
    accepted_loans = pred_df[pred_df['pred_loan_status'] == 0]
    num_default_loans.append(len(accepted_loans[accepted_loans['loan_status'] == 1]))
    num_non_default_loans.append(len(accepted_loans[accepted_loans['loan_status'] == 0]))
    bad_rates.append(np.sum((accepted_loans['loan_status']) / len(accepted_loans['loan_status'])).round(3))
    num_accepted_loans.append(accepted_loans['loan_status'].count())
    est_val.append((len(accepted_loans[accepted_loans['loan_status'] == 0])) -
                   (len(accepted_loans[accepted_loans['loan_status'] == 1])) * 
                   X_test['loan_amnt'].mean())
    
strat_df = pd.DataFrame(zip(accept_rates, thresholds, bad_rates, num_accepted_loans, num_default_loans, num_non_default_loans, est_val),
                        columns = ['Acceptance Rate','Threshold','Bad Rate','Num Accepted Loans','No Default', 'No Non Default','Estimated Value'])
strat_df[['Acceptance Rate','Threshold','Bad Rate']].boxplot()
plt.show()
print(strat_df.loc[strat_df['Estimated Value'] == np.max(strat_df['Estimated Value'])])

# optimum threshold = 0.80

# Calculate the bank's expected loss on the defaults
pred_df['loss_given_default'] = 1
pred_df['loan_amnt'] = np.array(X_test['loan_amnt'])
pred_df['expected_loss'] = pred_df['y_pred'] * pred_df['loss_given_default'] * pred_df['loan_amnt']
tot_exp_loss = round(np.sum(pred_df['expected_loss']),2)
print('Total expected loss: ', '${:,.2f}'.format(tot_exp_loss))



# pickle model
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train, np.ravel(y_train))

import pickle
pickl = {'model': model}
pickle.dump(pickl, open('model_file' + ".p", "wb" ))

file_name = 'model_file.p'
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
