import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
#%matplotlib qt

# Time the code
start_time = time.time()

# Check columns for numerical/categorical replace with mean/median
class FillNA:
    def fill_median_na(self, NA_columns, data):
        for entries in NA_columns:
            try:
                data[entries]=data[entries].fillna(np.median(data[entries]))   
            except TypeError:
                data=data[data[entries].notna()]
            return data
    def remove_na_columns(self, NA_columns, data):
        for entries in NA_columns:
            try:
                data[entries]=data[entries].fillna(np.mean(data[entries])) 
            except TypeError:
                data=data[data[entries].notna()]            
        finaldata=data
        return finaldata
                        
class BestFeat:
    def __init__(self, X, y):
        self.X=X
        self.y=y
    def CorrelationMap(self):
        combined_df=pd.concat([self.X,self.y], axis=1)
        sns.heatmap(combined_df)
    def Params_1(self, model, col_names):
        mod=model.fit(self.X, self.y)
        coef = pd.Series(mod.feature_importances_, index=col_names)
        return coef
    def Params_2(self, model, col_names):
        mod=model.fit(self.X, self.y)
        coef = pd.Series(mod.coef_, index=col_names)
        best_features=coef.sort_values()
        print("The best score is:", mod.score(self.X, self.y))
        return best_features

# Under specified NaN percentage keep the column
def MissingVals(PercentMissing, columns, pct):
    for i in range(len(PercentMissing)):
        if PercentMissing[i][1]<pct:
            pass
        else:
            columns.remove(PercentMissing[i][0])
    return columns

# Obtain columns with numerical values
def numerical_columns(columns):
    numerical_entries=pd.DataFrame()
    for col in columns.columns:
        try:
            if columns[col].sum()>=0:
                numerical_entries=pd.concat([numerical_entries, columns[col]], axis=1)
            else:
                pass
        except TypeError:
            pass
    return numerical_entries

# Remove the n largest valued rows for a specific column
def n_largest_removal(dataframe, column, n):
    largest_vals=list(dataframe.nlargest(n, column).index)
    dataframe=dataframe.drop(largest_vals)
    return dataframe

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
removed_na=FillNA().remove_na_columns(na_columns, data)
removed_na=removed_na.drop(["Id"], axis=1)
removed_na=removed_na.dropna(axis=1)

initial_encoding_val=65
numbered_categorical_recording={}
for i in range(len(removed_na)):
    if str(removed_na.iloc[i, 0]) in numbered_categorical_recording:
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]
    else:
        numbered_categorical_recording[str(removed_na.iloc[i, 0])]=chr(initial_encoding_val)
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]
        initial_encoding_val+=1

# Boxplots to visualize our numerical values for LotArea,GrLivArea remove the 4 highest points
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_na["LotArea"])
axes[1].boxplot(removed_na["GrLivArea"])
plt.show()

# Remove 4 highest points for Lot Area
removed_area_outliers=n_largest_removal(removed_na, "LotArea", 4)

# Replot box plots again
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_area_outliers["LotArea"])
axes[1].boxplot(removed_area_outliers["GrLivArea"])
plt.show()

# Remove the highest points for Group Living Area
removed_area_outliers=n_largest_removal(removed_area_outliers, "GrLivArea", 4)

# Replot box plots again
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_area_outliers["LotArea"])
axes[1].boxplot(removed_area_outliers["GrLivArea"])
plt.show()

# Separate true prediction values from the dataset
y=removed_area_outliers.iloc[:, -1]
removed_area_outliers=removed_area_outliers.drop(y.name ,axis=1)
# Record the columns which are numerical
all_numerical=numerical_columns(removed_area_outliers).columns

# One-hot encode the categorical features
encoded_features=pd.get_dummies(removed_area_outliers)
encoded_cols=encoded_features.columns

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_features, y, test_size=0.2, random_state=42)

# Take the numerical columns and scale, separate from rest of dataset
numerical_train_col=X_train[all_numerical]
categorical_train_predictors=X_train.drop(numerical_train_col.columns ,axis=1).reset_index().drop(["index"], axis=1)
numerical_test_col=X_test[all_numerical]
categorical_test_predictors=X_test.drop(numerical_test_col.columns, axis=1).reset_index().drop(["index"], axis=1)

# Reset indices for Sales Price
y_train=y_train.reset_index().drop(["index"], axis=1)
y_test=y_test.reset_index().drop(["index"], axis=1)

# Scale the numerical features
standard_scaling=StandardScaler()
encoded_features_train=pd.DataFrame(standard_scaling.fit_transform(numerical_train_col), columns=numerical_train_col.columns)
encoded_features_test=pd.DataFrame(standard_scaling.fit_transform(numerical_test_col), columns=numerical_test_col.columns)

# Merge the one-hot encoded predictors and the standardized predictors
encoded_features_train=pd.concat([categorical_train_predictors, encoded_features_train] ,axis=1)
encoded_features_test=pd.concat([encoded_features_test, categorical_test_predictors] ,axis=1)

# Variable Selection
weighted_col_names=encoded_features_train.columns
feature_strength=BestFeat(encoded_features_train, y_train).Params_2(LassoCV(cv=5), weighted_col_names)
largest_absolute_strength=abs(feature_strength)

non_zero_features=largest_absolute_strength[abs(largest_absolute_strength)>5000]
strongest_features=non_zero_features.sort_values()

# Visualize the features
index_names=[strongest_features.index[i] for i in range(len(strongest_features))]
strongest_feature_vals=[strongest_features[i] for i in range(len(strongest_features))]
plt.figure(figsize=(30,15))
plt.title("Important Features")
plt.xticks(rotation='vertical')
plt.bar(index_names, strongest_feature_vals)
plt.show()

# Check for high amounts of correlation with a heatmap
strongest_predictors=encoded_features_train[strongest_features.index]
correlation_map=pd.concat([strongest_predictors, y_train], axis=1)
plt.figure(figsize=(30,15))
sns.heatmap(correlation_map.corr(), annot=True, square=True, annot_kws={"fontsize":4})
plt.show()

# Choose the largest weighted parameters
best_train_features=strongest_predictors
best_test_features=encoded_features_test[strongest_features.index]

