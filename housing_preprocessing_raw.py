import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
import random

random.seed(42)
          
# Functions/Classes that will be used to filter features
class BestFeat:
    def Params(self, predictors, output, model, col_names):
        mod=model.fit(predictors, output)
        coef = pd.Series(mod.coef_, index=col_names)
        best_features=coef.sort_values()
        return best_features
    def Trees(self, predictors, output, model, col_names):
        mod=model.fit(predictors, output)
        coef = pd.Series(mod.feature_importances_, index=col_names)
        best_features=coef.sort_values()
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
imputed_data=data.copy()
# Search for columns with NA and give percentage of columns NA
na_checking=(data.isna().sum())/len(data)
na_missing=[]
na_columns=[]

count=0
# Separate all the NA columns from columns with no missing values
for i in na_checking:
    if i>0:
        na_missing.append((data.columns[count], i))
        na_columns.append(data.columns[count])
    count+=1

# Remove columns with greater than 20% of values missing
na_columns=MissingVals(na_missing, na_columns, 0.2)

# Separate categorical from numerical columns
categorical_only=[]
numerical_only=na_columns.copy()
for i in na_columns:
    if type(data[i][0])==str:
        categorical_only.append(i)
        numerical_only.remove(i)

# Count the available values for each categorical column of NaN values
categorical_entry_count=[data[cols].value_counts() for cols in categorical_only]
# Impute categorical nan with most frequent category if high amount, otherwise randomly choose
imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for i,j in enumerate(categorical_entry_count):
    cat_col_name=categorical_only[i]
    ratio_majority=j[0]/sum(j)
    if ratio_majority>0.66:
        cat_col=data[cat_col_name].values.reshape(-1,1)
        most_freq=imp_freq.fit_transform(cat_col)
        imputed_data[cat_col_name]=most_freq
    else:
        choices=j.index
        prob=[k/sum(categorical_entry_count[i]) for k in categorical_entry_count[i]]
        imputed_data[cat_col_name]=data[cat_col_name].apply(lambda x: np.random.choice(choices, p=prob) if pd.isnull(x) else x)

# Retrieve variables most correlated to the columns with missing values
numerical_corr=[data.corr()[i].drop(index='SalePrice').sort_values(ascending=False).apply(lambda x: x if (x < 1 and x>0.3) else np.nan) for i in numerical_only]
numerical_corr=[i.dropna() for i in numerical_corr]

# Standardize data to apply KNN and impute values
standardize_imputer=StandardScaler()

# Impute missing numerical values by applying KNearest Neighbors
knn_impute=KNNImputer(missing_values=np.nan, n_neighbors=10)
for i in numerical_corr:
    imputed_col_name=i.name
    columns=i.index.append(pd.Index([imputed_col_name]))
    standardized_numerical_na=standardize_imputer.fit_transform(data[columns])
    tmp=knn_impute.fit_transform(standardized_numerical_na)
    transformed=standardize_imputer.inverse_transform(tmp)
    imputed_data[imputed_col_name]=transformed[:,transformed.shape[1]-1]
    
# Fill numerical values with the average value, drop ID, encode dwelling classification
removed_na=imputed_data.drop(["Id"], axis=1)
removed_na=removed_na.dropna(axis=1)

# Convert MSubClass into categorical, rather than keeping in numerical form
initial_encoding_val=65
numbered_categorical_recording={}
for i in range(len(removed_na)):
    if str(removed_na.iloc[i, 0]) in numbered_categorical_recording:
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]
    else:
        numbered_categorical_recording[str(removed_na.iloc[i, 0])]=chr(initial_encoding_val)
        removed_na.iloc[i, 0]=numbered_categorical_recording[str(removed_na.iloc[i, 0])]
        initial_encoding_val+=1

# Remove 4 highest points for Lot Area
removed_area_outliers=n_largest_removal(removed_na, "LotArea", 4)

# Remove the highest points for Group Living Area
removed_area_outliers=n_largest_removal(removed_area_outliers, "GrLivArea", 4)

# Combine columns for possible predictors
removed_area_outliers['Bathrooms']=removed_area_outliers['BsmtFullBath'] + removed_area_outliers['FullBath'] + 0.5*removed_area_outliers['BsmtHalfBath'] +  0.5*removed_area_outliers['HalfBath']
removed_area_outliers['TotalInsideArea']=removed_area_outliers['GrLivArea'] + removed_area_outliers['TotalBsmtSF']

# One-hot encode the categorical features
full_untransformed_data=pd.get_dummies(removed_area_outliers)

# Separate true prediction values from the dataset, high variance for Sales Price so apply log transform to Sale Price
unscaled_y=full_untransformed_data.loc[:, 'SalePrice']
y=np.log(full_untransformed_data.loc[:, 'SalePrice'])
features_only=full_untransformed_data.drop(y.name ,axis=1)

# Record the columns which are numerical
all_numerical=numerical_columns(removed_area_outliers).columns

# Separate categorical features
full_categorical=full_untransformed_data.drop(features_only ,axis=1).reset_index().drop(["index"], axis=1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_only, y, test_size=0.2, random_state=42)

# Scale the training set features
standard_scaling=StandardScaler()
full_transformed_data_train=pd.DataFrame(standard_scaling.fit_transform(X_train), columns=X_train.columns)

# Obtain the entire full training data
whole_training=pd.concat([X_train, y_train], axis=1)

# Variable selection based upon highest correlations with Sales Price
sales_price_corr=abs(whole_training.corr()["SalePrice"]).sort_values().drop(index="SalePrice")
strongest_corr_feat=sales_price_corr[sales_price_corr>0.65].drop(["GarageCars", "GrLivArea"])

# Variable Selection with Lasso
weighted_col_names=full_transformed_data_train.columns
feature_strength=BestFeat().Params(full_transformed_data_train, y_train ,LassoCV(cv=5, random_state=42), weighted_col_names)
largest_absolute_strength=abs(feature_strength)

# Select Lasso Features
strongest_lasso_feat=largest_absolute_strength.nlargest(8)
strongest_features=strongest_lasso_feat.sort_values()

# Not the highest correlation, will compare in visualization clustering which is better

# Visualize the features
corr_index_names=[strongest_corr_feat.index[i] for i in range(len(strongest_corr_feat))]
lasso_index_names=[strongest_features.index[i] for i in range(len(strongest_features))]
strongest_lasso_vals=[strongest_features[i] for i in strongest_features.index]
strongest_corr_vals=[strongest_corr_feat[i] for i in strongest_corr_feat.index]

# Drop GrLivArea due to repition and high correlation.
lasso_index_names.remove("GrLivArea")

# Combined best features index
best_columns=list(set(corr_index_names+lasso_index_names))
