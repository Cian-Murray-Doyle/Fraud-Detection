# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:07:30 2024

@author: cian3
"""

#Importing nescessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


#Importing the dataset and creating dataframe
file_path = r"C:\Users\cian3\OneDrive\Documents"
file_name = r"\creditcard.csv"

credit_fraud = pd.read_csv(file_path+file_name)
credit_fraud.head()

#Examine dataframe
print('Dataframe Described','\n',credit_fraud.describe())
print('Total Number of NULL Values:',sum(credit_fraud.isna().sum(axis=0)))

#Gaining a view of data split
print('Fraud accounts for ',credit_fraud['Class'].value_counts()[1]*100/len(
    credit_fraud),"% of dataset.")

sns.countplot(x='Class', data=credit_fraud, palette=['green', 'red'])
plt.title('Distribution of Fraud Transactions')
plt.xticks([0, 1], ['Genuine', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#Showing that amount and time need to be standardised, inline with other features
fig, ax = plt.subplots(1, 2, figsize=(18,4))
amount_val = credit_fraud['Amount'].values
time_val = credit_fraud['Time'].values

sns.distplot(amount_val, ax=ax[0], color='b')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

#Standardising Amount and Time
std_scaler = StandardScaler()

credit_fraud['scaled_amount'] = std_scaler.fit_transform(credit_fraud['Amount'].values.reshape(-1,1))
credit_fraud['scaled_time'] = std_scaler.fit_transform(credit_fraud['Time'].values.reshape(-1,1))

#Removing original rows, keeping standardised rows
credit_fraud.drop(['Time','Amount'], axis=1, inplace=True)

credit_fraud.head()

#Creating a 50/50 fraud/geniuine dataset
fraud = credit_fraud.loc[credit_fraud['Class']==1]
genuine = credit_fraud.loc[credit_fraud['Class']==0][:len(fraud)]

normal_fraud = pd.concat([fraud,genuine])

#Shuffling again, helps avoid overfitting model
fraud_c = normal_fraud.sample(frac=1, random_state=42)


sns.countplot(x='Class',data=fraud_c,palette=['green', 'red'])
plt.title('Distribution of Fraud Transactions')
plt.xticks([0, 1], ['Genuine', 'Fraud'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#Can now see correlation of features clearly
correlation_matrix = fraud_c.corr()
plt.figure(figsize=(10, 8))  # Optionally adjust the figure size
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

#Visualising Outliers
#Highly Negative Correlated Features
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V17', data=fraud_c,ax=axes[0], palette=['green','red'])
axes[0].set_title('V17 vs Class Negative Correlation')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V14', data=fraud_c,ax=axes[1], palette=['green','red'])
axes[1].set_title('V14 vs Class Negative Correlation')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V12', data=fraud_c,ax=axes[2], palette=['green','red'])
axes[2].set_title('V12 vs Class Negative Correlation')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V10', data=fraud_c,ax=axes[3], palette=['green','red'])
axes[3].set_title('V10 vs Class Negative Correlation')
axes[3].set_xticks([0, 1])
axes[3].set_xticklabels(['Genuine', 'Fraud'])

plt.show()

#Highly Positive Correlated Features
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V11', data=fraud_c,ax=axes[0], palette=['green','red'])
axes[0].set_title('V11 vs Class Positive Correlation')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V4', data=fraud_c,ax=axes[1], palette=['green','red'])
axes[1].set_title('V4 vs Class Positive Correlation')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V2', data=fraud_c,ax=axes[2], palette=['green','red'])
axes[2].set_title('V2 vs Class Positive Correlation')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Genuine', 'Fraud'])

sns.boxplot(x='Class', y='V19', data=fraud_c,ax=axes[3], palette=['green','red'])
axes[3].set_title('V19 vs Class Positive Correlation')
axes[3].set_xticks([0, 1])
axes[3].set_xticklabels(['Genuine', 'Fraud'])

plt.show()

# Distribution of fraud transactions
f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,6))

v14_fraud_dist = fraud_c['V14'].loc[fraud_c['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = fraud_c['V12'].loc[fraud_c['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = fraud_c['V10'].loc[fraud_c['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()

#Removing outliers IQR method
v14 = fraud_c['V14'].loc[fraud_c['Class']==1].values
q1 = np.percentile(v14,25)
q3 = np.percentile(v14,75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5* iqr
fraud_c = fraud_c[(fraud_c['V14'] > lower_bound)&(fraud_c['V14'] < upper_bound)]

v12 = fraud_c['V12'].loc[fraud_c['Class']==1].values
q1 = np.percentile(v12,25)
q3 = np.percentile(v12,75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5* iqr
fraud_c = fraud_c[(fraud_c['V12'] > lower_bound)&(fraud_c['V12'] < upper_bound)]

v10 = fraud_c['V10'].loc[fraud_c['Class']==1].values
q1 = np.percentile(v10,25)
q3 = np.percentile(v10,75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr 
upper_bound = q3 + 1.5* iqr
fraud_c = fraud_c[(fraud_c['V10'] > lower_bound)&(fraud_c['V10'] < upper_bound)]

#Test and Train datasets
predictors = fraud_c.drop('Class',axis=1)
response = fraud_c['Class']

predictors_train, predictors_test, response_train, response_test = train_test_split(
    predictors, response, test_size=0.2, random_state=42)

#Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(predictors_train,response_train)
predictions = logistic_regression.predict(predictors_test)

print('Logistic Regression Report:','\n',classification_report(response_test,predictions,target_names=['Genuine','Fraud']))

#Confusion Matrix to visualise accuracy
cm= confusion_matrix(response_test,predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="flare")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5, 1.5], ['Genuine', 'Fraud'])
plt.yticks([0.5, 1.5], ['Genuine', 'Fraud'])
plt.show()

#Forecasting
#Time column is stored as seconds elapsed from first transaction
#Dataset is from September 2013 - Assuming start date is 1st
start_date = pd.to_datetime('2020-09-01')
credit_fraud_for = pd.read_csv(file_path+file_name)
credit_fraud_for['date_time'] = pd.to_timedelta(credit_fraud_for['Time'],
                                           unit='s') + start_date
hourly_values = credit_fraud_for.resample('H', on='date_time').Amount.sum()
hourly_transaction_count = credit_fraud_for.resample('H', on='date_time').size()

#Visualising transaction value over time
plt.figure(figsize=(14, 7))
plt.plot(hourly_values.index, hourly_values, label='Hourly Transaction Amounts', marker='o')
plt.title('Hourly Transaction Amounts Over Two Days')
plt.xlabel('Hour')
plt.ylabel('Transaction Amount')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#Visualising transaction count over time
plt.figure(figsize=(14, 7))
plt.plot(hourly_transaction_count.index, hourly_transaction_count, label='Hourly Transaction Count', marker='x')
plt.title('Hourly Transaction Count Over Two Days')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#Finding ideal parameters for model
res_grid = smt.arma_order_select_ic(hourly_values, max_ar = 5, max_ma = 5, ic=['aic','bic'])

print('AIC')
print(res_grid.aic)
print('BIC')
print(res_grid.bic)

print(res_grid.aic_min_order)
print(res_grid.bic_min_order)

#Time frame is limited so will use first day to train and second day to test
hourly_values.index = pd.DatetimeIndex(hourly_values.index.values,
                                       freq= hourly_values.index.inferred_freq)
train = hourly_values.iloc[:-24]
test = hourly_values.iloc[-24:]

#Building and forecasting with model
model = ARIMA(train, order=(2,0,5))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=24)

print(fitted_model.summary())

#Visualising accuracy
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label = 'Actual Value')
plt.plot(test.index, forecast, label='Forecasted Value', color='red')
plt.title('Hourly Transaction Value Forecast')
plt.xlabel('Time')
plt.ylabel('Transaction Amount')
plt.legend()
plt.xticks(rotation = 45)
plt.show()

#Confirming innaccuracy with RMSE
rmse = np.sqrt(mean_squared_error(test,forecast))
print("Root Mean Squared Error: ",rmse)