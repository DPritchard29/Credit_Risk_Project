# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:28:31 2022

@author: Dan
"""

import pandas as pd
import matplotlib.pyplot as plt

cr_loan = pd.read_csv('cr_loan2.csv')

# Explore credit data
cr_loan.dtypes
cr_loan.head()

# visualisations
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()

plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()

pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True)
pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']])

pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean')

cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()

# Deal with missing data
cr_loan.columns[cr_loan.isnull().any()]
cr_loan[cr_loan['person_emp_length'].isnull()].head()
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()

cr_loan['loan_int_rate'].isnull().sum()
null_cols = cr_loan[cr_loan['loan_int_rate'].isnull()].index
cr_loan.drop(null_cols, inplace = True)

# Looking into outliers
pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
            values=cr_loan['person_emp_length'], aggfunc='max')

plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()

# Remove employment length and age outliers
emp_outliers = cr_loan[cr_loan['person_emp_length'] > 60].index
age_outliers = cr_loan[cr_loan['person_age'] > 100].index
cr_loan = cr_loan.drop(emp_outliers, age_outliers).drop(age_outliers)

pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
            values=cr_loan['person_emp_length'], aggfunc=['min','max'])

plt.scatter(cr_loan['person_age'], cr_loan['loan_int_rate'],
            c = cr_loan['loan_status'],
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()


cr_loan.to_csv('cr_loan_clean.csv', index = False)