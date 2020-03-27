import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check columns for numerical/categorical replace with mean/median
class FillNA:
    def __init__(self, NA_columns, data):
        self.na=NA_columns
        self.data=data
    def fill_median_na(self):
        for entries in self.na:
            try:
                self.data[entries]=self.data[entries].fillna(np.median(self.data[entries]))   
            except TypeError:
                pass
    def remove_na_columns(self):
        for entries in self.na:
            try:
                self.data[entries]=self.data[entries].fillna(np.mean(self.data[entries])) 
            except TypeError:
                self.data=self.data[self.data[entries].notna()]
            
        finaldata=self.data
        return finaldata
                        
class BestFeat:
    def __init__(self, X, y):
        self.X=X
        self.y=y
    def Params(self, model):
        mod=model.fit(self.X, self.y)
        params=mod.feature_importances_
        return params

def MissingVals(PercentMissing, columns, pct):
    for i in range(len(PercentMissing)):
        if PercentMissing[i][1]<pct:
            pass
        else:
            columns.remove(PercentMissing[i][0])
            
    return columns
    

# Check for columns with NaN vaues
data=pd.read_csv("train.csv")

na_checking=(data.isna().sum())/len(data)
na_missing=[]
na_columns=[]

count=0
for i in na_checking:
    if i>0:
        na_missing.append((data.columns[count], i))
        na_columns.append(data.columns[count])
    count+=1
    
na_columns=MissingVals(na_missing, na_columns, 0.2)
    
# Fill numerical values with the average value, drop ID, encode dwelling classification
removed_na=FillNA(na_columns, data).remove_na_columns()
removed_na=removed_na.drop(["Id"], axis=1)

initial_encoding_val=65
numbered_categorical_recording={}
for i in range(len(removed_na)):
    if str(removed_na.iloc[i, 0]) in numbered_categorical_recording:
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]
    else:
        numbered_categorical_recording[str(removed_na.iloc[i, 0])]=chr(initial_encoding_val)
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]

        initial_encoding_val+=1
   
# Feature Selection
standard_scaling=StandardScaler()
feature_count=len(removed_na.columns)-1
X=removed_na.iloc[:, :feature_count]
y=removed_na.iloc[:, -1]

encoded_features=pd.get_dummies(X)
encoded_cols=encoded_features.columns

encoded_features=standard_scaling.fit_transform(encoded_features)

encoded_col_values=[]
for i in range(len(encoded_features[0])):
    encoded_col_values.append(encoded_features[:,i])
    
encoded_df=pd.DataFrame()
count=0
for i in encoded_cols:
    encoded_df[i]=encoded_col_values[count]
    count+=1

encoded_df.columns=encoded_cols

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf=Lasso()
rf.fit(encoded_df, y)
len(rf.feature_importances_)
BestFeat(encoded_df, y).Params(RandomForestRegressor())

   
# Variable Selection