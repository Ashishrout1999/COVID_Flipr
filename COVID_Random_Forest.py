from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np


#Fetching the training dataset
import pandas as pd
train_data = pd.read_excel('E:/6th_Semester/ML/Flipr/Trying/Train_dataset(1).xlsx', 'Train_dataset')
test_data = pd.read_excel('E:/6th_Semester/ML/Flipr/Trying/Test_dataset_1.xlsx')
test_27March = pd.read_excel('E:/6th_Semester/ML/Flipr/Trying/Train_dataset(1).xlsx', 'Train_27March')

#Checking for missing values
#print(train_data.isnull().sum())

#Imputing values through simple method like median of the column
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(train_data[['Children','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']])
X = pd.DataFrame(imp.transform(train_data[['Children','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']]))
train_data[['Children','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']] = imp.transform(train_data[['Children','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']])

# Imputing categorical data by mode 
imp_string = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_string.fit(train_data[['Occupation','Mode_transport','comorbidity','cardiological_pressure']])
Z = pd.DataFrame(imp_string.transform(train_data[['Occupation','Mode_transport','comorbidity','cardiological_pressure']]))
train_data[['Occupation','Mode_transport','comorbidity','cardiological_pressure']] = imp_string.transform(train_data[['Occupation','Mode_transport','comorbidity','cardiological_pressure']])

#Imputing a constant value for unknown names
imp_name = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = "Unknown")
N = pd.DataFrame(imp_name.fit_transform(train_data['Name'].values.reshape(-1,1)))
train_data['Name'] = imp_name.fit_transform(train_data['Name'].values.reshape(-1,1))

#Rechecking on whether there is any more missing data
#print(train_data.isnull().sum())

Description = train_data.describe(include = "all")

#To test correlation between numeric features. Also correlation between categorical and numeric features were carried out in excel. 
A = ['Children','cases/1M','Deaths/1M','Age','Coma score','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','Insurance','salary','FT/month']

for i in A:
    for j in A:
        if (j is not i):
            corr,_ = pearsonr(train_data[i],train_data[j])
            if corr > 0.3:
                print(corr,i,j)

#It was observed that there is no such strong correlation between independent features apart from correlation between cases/1M and Death/1M.

#Processing the categorical data
B = ['Married','Occupation','Mode_transport','comorbidity','Pulmonary_score','cardiological_pressure']
train_data = pd.concat((train_data,pd.get_dummies(train_data.Married)),1)
train_data = pd.concat((train_data,pd.get_dummies(train_data.Occupation)),1)
train_data = pd.concat((train_data,pd.get_dummies(train_data.Mode_transport)),1)
train_data = pd.concat((train_data,pd.get_dummies(train_data.comorbidity)),1)
train_data = pd.concat((train_data,pd.get_dummies(train_data.Pulmonary_score)),1)
train_data = pd.concat((train_data,pd.get_dummies(train_data.cardiological_pressure)),1)

#Selecting names of feature columns 
cols = list(train_data.columns)
unwanted = [0,1,2,3,4,5,7,8,11,14,15,27]
for ele in sorted(unwanted, reverse = True):  
    del cols[ele]



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data.loc[:,cols] = sc.fit_transform(train_data.loc[:,cols])

#Use of iloc could have simplified column selection.


#Training data (features I and output y)
I = train_data.loc[:,cols].values
y = train_data.loc[:,['Infect_Prob']].values

"""
# Basic model

from sklearn.model_selection import train_test_split
I_train, I_test, y_train, y_test = train_test_split(I, y, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(I_train, y_train)
y_pred = regressor.predict(I_test)
print(y_pred)



from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
       
"""
# Model with cross-validation

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

parameter_grid = {
    'max_depth': [3,5],
    'n_estimators': [200,400],
    'min_samples_leaf': [4,6],
    'min_samples_split': [8,10]
}
abc = RandomForestRegressor()
grid_search = GridSearchCV(estimator = abc, param_grid = parameter_grid, 
                          cv = 5,scoring = 'neg_mean_squared_error', n_jobs = -1, verbose = 2)
grid_result = grid_search.fit(I, y)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_
print(best_parameters)

best_grid = grid_search.best_estimator_
#print(best_grid)

y_pred = best_grid.predict(I)
print(y_pred)

MSE = cross_val_score(best_grid, I, y, cv=5, scoring='neg_mean_squared_error')
print(MSE)

RMSE = np.sqrt(-np.average(MSE))
print('RMSE is ',RMSE)

MAR = cross_val_score(best_grid, I, y, cv=5, scoring='neg_mean_absolute_error')
print(MAR)


# Preparation of test_data to be input into the model


#Checking for missing values
#print(test_data.isnull().sum())

#Imputing a constant value for unknown names
imp_name = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = "XYZ")
N = pd.DataFrame(imp_name.fit_transform(test_data['Name'].values.reshape(-1,1)))
test_data['Name'] = imp_name.fit_transform(test_data['Name'].values.reshape(-1,1))

#print(test_data.isnull().sum())

# Processing categorical data
test_data = pd.concat((test_data,pd.get_dummies(test_data.Married)),1)
test_data = pd.concat((test_data,pd.get_dummies(test_data.Occupation)),1)
test_data = pd.concat((test_data,pd.get_dummies(test_data.Mode_transport)),1)
test_data = pd.concat((test_data,pd.get_dummies(test_data.comorbidity)),1)
test_data = pd.concat((test_data,pd.get_dummies(test_data.Pulmonary_score)),1)
test_data = pd.concat((test_data,pd.get_dummies(test_data.cardiological_pressure)),1)

#Selecting names of feature columns 
test_cols = list(test_data.columns)
unwanted = [0,1,2,3,4,5,7,8,11,14,15]
for ele in sorted(unwanted, reverse = True):  
    del test_cols[ele]



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_data.loc[:,test_cols] = sc.fit_transform(test_data.loc[:,test_cols])


#Test data (input features I_test)
I_test = test_data.loc[:,test_cols].values

Test_data_predict = best_grid.predict(I_test)
print(Test_data_predict)


# Preparation of data for prediction on 27 March (After updating diuresis value)


#Imputing values through simple method like median of the column
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(test_27March[['Children','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']])
X = pd.DataFrame(imp.transform(test_27March[['Children','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']]))
test_27March[['Children','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']] = imp.transform(test_27March[['Children','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Insurance','FT/month']])

# Imputing categorical data by mode 
imp_string = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_string.fit(test_27March[['Occupation','Mode_transport','comorbidity','cardiological_pressure']])
Z = pd.DataFrame(imp_string.transform(test_27March[['Occupation','Mode_transport','comorbidity','cardiological_pressure']]))
test_27March[['Occupation','Mode_transport','comorbidity','cardiological_pressure']] = imp_string.transform(test_27March[['Occupation','Mode_transport','comorbidity','cardiological_pressure']])

#Imputing a constant value for unknown names
imp_name = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = "Unknown")
N = pd.DataFrame(imp_name.fit_transform(test_27March['Name'].values.reshape(-1,1)))
test_27March['Name'] = imp_name.fit_transform(test_27March['Name'].values.reshape(-1,1))


test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.Married)),1)
test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.Occupation)),1)
test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.Mode_transport)),1)
test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.comorbidity)),1)
test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.Pulmonary_score)),1)
test_27March = pd.concat((test_27March,pd.get_dummies(test_27March.cardiological_pressure)),1)

#Selecting names of feature columns 
test_cols_27 = list(test_27March.columns)
unwanted = [0,1,2,3,4,5,7,8,11,14,15]
for ele in sorted(unwanted, reverse = True):  
    del test_cols_27[ele]



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
test_27March.loc[:,test_cols_27] = sc.fit_transform(test_27March.loc[:,test_cols_27])


#Test data (input features I_test_27)
I_test_27 = test_27March.loc[:,test_cols_27].values

Test_data_predict_27 = best_grid.predict(I_test_27)
print(Test_data_predict_27)













