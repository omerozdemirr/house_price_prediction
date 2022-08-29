import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder              
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from xgboost import XGBRegressor


d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train.isna().sum().sort_values(ascending=False)

#total missing values

df = d_train.isna().sum().sum()
print(df)

#total missing values in percentage 
print(len(d_train)/df*100)


#filling the columns below with 'None' values

d_train['PoolQC']= d_train['PoolQC'].fillna('None')
d_train['MiscFeature']= d_train['MiscFeature'].fillna('None')
d_train['Alley']= d_train['Alley'].fillna('None')
d_train['Fence']= d_train['Fence'].fillna('None')
d_train['FireplaceQu']= d_train['FireplaceQu'].fillna('None')

#impute lotfrotage column with its mean values
d_train['LotFrontage'] =d_train['LotFrontage'].fillna(d_train['LotFrontage'].mean())


#filling these columns with zero values
d_train['GarageYrBlt'] = d_train['GarageYrBlt'].fillna(0)
d_train['MasVnrArea'] = d_train['MasVnrArea'].fillna(0)

#filling the missing data with miost frequent values]

d_train['GarageCond'] = d_train['GarageCond'].fillna(d_train['GarageCond'].value_counts().idxmax())
d_train['GarageType'] = d_train['GarageType'].fillna(d_train['GarageType'].value_counts().idxmax())
d_train['GarageFinish'] = d_train['GarageFinish'].fillna(d_train['GarageFinish'].value_counts().idxmax())
d_train['GarageQual'] = d_train['GarageQual'].fillna(d_train['GarageQual'].value_counts().idxmax())
d_train['BsmtFinType2'] = d_train['BsmtFinType2'].fillna(d_train['BsmtFinType2'].value_counts().idxmax())
d_train['BsmtExposure'] = d_train['BsmtExposure'].fillna(d_train['BsmtExposure'].value_counts().idxmax())
d_train['BsmtQual'] = d_train['BsmtQual'].fillna(d_train['BsmtQual'].value_counts().idxmax())
d_train['BsmtCond'] = d_train['BsmtCond'].fillna(d_train['BsmtCond'].value_counts().idxmax())
d_train['BsmtFinType1'] = d_train['BsmtFinType1'].fillna(d_train['BsmtFinType1'].value_counts().idxmax())
d_train['MasVnrType'] = d_train['MasVnrType'].fillna(d_train['MasVnrType'].value_counts().idxmax())
d_train['Electrical'] = d_train['Electrical'].fillna(d_train['Electrical'].value_counts().idxmax())

#lets check if we have any missed values 
print(d_train.isna().sum().sort_values(ascending =False))


#defining how required variables correlate to our target variable
corr = d_train.corr()
corr.sort_values(['SalePrice'],ascending =False,inplace =True)
print(corr.SalePrice)

#label Encoding 

label = LabelEncoder()
for i in d_train.columns:
    if d_train[i].dtypes ==object:
        d_train[i] = label.fit_transform(d_train[i])

#selecting the features 
X =d_train.drop('SalePrice',axis =1)
y =d_train['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size =0.3,random_state=42)

#creating model with different functions 
'''
def decision_tree_model(X_train, y_train):
    
    tree= DecisionTreeRegressor(random_state=1)
    tree.fit(X_train, y_train)
    y_prediction_tree = tree.predict(X_test)
    
    #metrics of decision tree regressor 
    
    MeanAbErr_tree = mean_absolute_error(y_test, y_prediction_tree)
    MeanSqErr_tree= metrics.mean_squared_error(y_test,y_prediction_tree)

    RootMeanSqErr_tree = np.sqrt(metrics.mean_squared_error(y_test, y_prediction_tree))
    
    
    print('Decision Tree: ',r2_score(y_test,y_prediction_tree))
    print('Mean Absolute Error: ', MeanAbErr_tree)
    print('Mean Square Error: ', MeanSqErr_tree)
    print('Root Mean Square Error: ', RootMeanSqErr_tree)
    
  '''
def xboos(X_train, y_train):
    xgboost =XGBRegressor()
    xgboost.fit(X_train,y_train)
    y_pred_xgboost =xgboost.predict(X_test)
d_test.isna().sum().sort_values(ascending = False)
    
d_test['PoolQC']= d_test['PoolQC'].fillna('None')
d_test['MiscFeature']= d_test['MiscFeature'].fillna('None')
d_test['Alley']= d_test['Alley'].fillna('None')
d_test['Fence']= d_test['Fence'].fillna('None')
d_test['FireplaceQu']= d_test['FireplaceQu'].fillna('None')

#impute Lotfrontofa colum with its mean values 
d_test['LotFrontage'] = d_test['LotFrontage'].fillna(d_test['LotFrontage'].mean())

#filling these columns with zero values 

d_test['GarageYrBlt']= d_test['GarageYrBlt'].fillna(0)
d_test['MasVnrArea']= d_test['MasVnrArea'].fillna(0)

# Filling the missing data with most_frequent values

d_test['GarageCond'] = d_test['GarageCond'].fillna(d_train['GarageCond'].value_counts().idxmax())
d_test['GarageType'] = d_test['GarageType'].fillna(d_train['GarageType'].value_counts().idxmax())
d_test['GarageFinish'] = d_test['GarageFinish'].fillna(d_train['GarageFinish'].value_counts().idxmax())
d_test['GarageQual'] = d_test['GarageQual'].fillna(d_train['GarageQual'].value_counts().idxmax())
d_test['BsmtFinType2'] = d_test['BsmtFinType2'].fillna(d_train['BsmtFinType2'].value_counts().idxmax())
d_test['BsmtExposure'] = d_test['BsmtExposure'].fillna(d_train['BsmtExposure'].value_counts().idxmax())
d_test['BsmtQual'] = d_test['BsmtQual'].fillna(d_train['BsmtQual'].value_counts().idxmax())
d_test['BsmtCond'] = d_test['BsmtCond'].fillna(d_train['BsmtCond'].value_counts().idxmax())
d_test['BsmtFinType1'] = d_test['BsmtFinType1'].fillna(d_train['BsmtFinType1'].value_counts().idxmax())
d_test['MasVnrType'] = d_test['MasVnrType'].fillna(d_train['MasVnrType'].value_counts().idxmax())
d_test['Electrical'] = d_test['Electrical'].fillna(d_train['Electrical'].value_counts().idxmax())
d_test['MSZoning'] = d_test['MSZoning'].fillna(d_train['MSZoning'].value_counts().idxmax())
d_test['BsmtFullBath'] = d_test['BsmtFullBath'].fillna(d_train['BsmtFullBath'].value_counts().idxmax())
d_test['BsmtHalfBath'] = d_test['BsmtHalfBath'].fillna(d_train['BsmtHalfBath'].value_counts().idxmax())
d_test['Functional'] = d_test['Functional'].fillna(d_train['Functional'].value_counts().idxmax())
d_test['Utilities'] = d_test['Utilities'].fillna(d_train['Utilities'].value_counts().idxmax())
d_test['Exterior2nd'] = d_test['Exterior2nd'].fillna(d_train['Exterior2nd'].value_counts().idxmax())
d_test['SaleType'] = d_test['SaleType'].fillna(d_train['SaleType'].value_counts().idxmax())
d_test['Exterior1st'] = d_test['Exterior1st'].fillna(d_train['Exterior1st'].value_counts().idxmax())
d_test['KitchenQual'] = d_test['KitchenQual'].fillna(d_train['KitchenQual'].value_counts().idxmax())
d_test['BsmtFinSF2'] = d_test['BsmtFinSF2'].fillna(d_train['BsmtFinSF2'].mean())
d_test['GarageArea'] = d_test['GarageArea'].fillna(d_train['GarageArea'].mean())
d_test['BsmtFinSF1'] = d_test['BsmtFinSF1'].fillna(d_train['BsmtFinSF1'].mean())
d_test['GarageCars'] = d_test['GarageCars'].fillna(d_train['GarageCars'].mean())
d_test['TotalBsmtSF'] = d_test['TotalBsmtSF'].fillna(d_train['TotalBsmtSF'].mean())
d_test['BsmtUnfSF'] = d_test['BsmtUnfSF'].fillna(d_train['BsmtUnfSF'].mean())


d_test.isnull().sum().sort_values(ascending=False)


label =LabelEncoder()
for x in d_test.columns:
    if d_test[x].dtypes == object:
        d_test[x] =label.fit_transform(d_test[x].astype(str))
        
    
    
    
xgboost =XGBRegressor()
xgboost.fit(X_train,y_train)
y_pred_xgboost =xgboost.predict(d_test)
print(y_pred_xgboost)


id_test = d_test['Id']
d_test =pd.DataFrame(d_test,columns =['ID'])
prediction =pd.DataFrame(y_pred_xgboost,columns = ['SalePrice'])

output =pd.concat([id_test,prediction],axis =1)
print(output)




output.to_csv('Submission.csv',index =False)
