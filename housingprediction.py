import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

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
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imp_mean.fit_transform(data[entries])
                data[entries]=data[data[entries].notna()]            
        finaldata=data
        return finaldata
                        
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
knn_impute=KNNImputer(missing_values=np.nan, n_neighbors=5)
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

# Boxplots to visualize our numerical values for LotArea,GrLivArea remove the 4 highest points
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_na["LotArea"])
axes[1].boxplot(removed_na["GrLivArea"])
axes[0].title.set_text("Lot Area")
axes[1].title.set_text("GrLivArea")
plt.show()

# Remove 4 highest points for Lot Area
removed_area_outliers=n_largest_removal(removed_na, "LotArea", 4)

# Replot box plots again
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_area_outliers["LotArea"])
axes[1].boxplot(removed_area_outliers["GrLivArea"])
axes[0].title.set_text("Lot Area")
axes[1].title.set_text("GrLivArea")
plt.show()

# Remove the highest points for Group Living Area
removed_area_outliers=n_largest_removal(removed_area_outliers, "GrLivArea", 4)

# Replot box plots again
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
axes[0].boxplot(removed_area_outliers["LotArea"])
axes[0].title.set_text("Lot Area")
axes[1].boxplot(removed_area_outliers["GrLivArea"])
axes[1].title.set_text("GrLivArea")
plt.show()

# Combine columns for possible predictors
removed_area_outliers['Bathrooms']=removed_area_outliers['BsmtFullBath'] + removed_area_outliers['FullBath'] + 0.5*removed_area_outliers['BsmtHalfBath'] +  0.5*removed_area_outliers['HalfBath']
removed_area_outliers['TotalInsideArea']=removed_area_outliers['GrLivArea'] + removed_area_outliers['TotalBsmtSF']

# Separate true prediction values from the dataset
y=removed_area_outliers.loc[:, 'SalePrice']
removed_area_outliers=removed_area_outliers.drop(y.name ,axis=1)
# Record the columns which are numerical
all_numerical=numerical_columns(removed_area_outliers).columns

# One-hot encode the categorical features
encoded_features=pd.get_dummies(removed_area_outliers)
encoded_cols=encoded_features.columns

# Standardize dataset for PCA and K-Means
full_categorical=encoded_features.drop(all_numerical ,axis=1).reset_index().drop(["index"], axis=1)
standardize_full_numerical=pd.DataFrame(StandardScaler().fit_transform(encoded_features[all_numerical]), columns=all_numerical)
standardize_full_categorical=pd.DataFrame(StandardScaler().fit_transform(full_categorical), columns=full_categorical.columns)
standardize_full=pd.concat([standardize_full_numerical, standardize_full_categorical], axis=1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_features, y, test_size=0.2, random_state=42)

# Take the numerical columns and scale, separate from rest of dataset
numerical_train_col=X_train[all_numerical]
categorical_train_predictors=X_train.drop(numerical_train_col.columns ,axis=1)
numerical_test_col=X_test[all_numerical]
categorical_test_predictors=X_test.drop(numerical_test_col.columns, axis=1)

# Merge the one-hot encoded predictors and the numerical predictors
complete_column_names=categorical_train_predictors.columns.append(numerical_train_col.columns)
features_train=pd.concat([categorical_train_predictors, numerical_train_col], axis=1).reset_index().drop(['index'], axis=1)
features_test=pd.concat([categorical_test_predictors, numerical_test_col], axis=1).reset_index().drop(['index'], axis=1)

# Scale the numerical features
standard_scaling=StandardScaler()
encoded_features_train=pd.DataFrame(standard_scaling.fit_transform(features_train), columns=complete_column_names)
encoded_features_test=pd.DataFrame(standard_scaling.fit_transform(features_test), columns=complete_column_names)

# Variable Selection with Lasso
weighted_col_names=encoded_features_train.columns
feature_strength=BestFeat().Params(encoded_features_train, y_train ,LassoCV(cv=5), weighted_col_names)
largest_absolute_strength=abs(feature_strength)

# Variable Selection with Random Forest
tree_features=BestFeat().Trees(encoded_features_train, y_train.values.ravel(), RandomForestRegressor(n_estimators=100), weighted_col_names)
largest_tree_strength=abs(feature_strength)

significant_lasso_features=largest_absolute_strength[abs(largest_absolute_strength)> 5000]
significant_tree_features=largest_tree_strength[abs(largest_tree_strength) > 5000]
strongest_features=significant_lasso_features.sort_values()
strongest_features_trees=significant_tree_features.sort_values()

# Visualize the features
tree_index_names=[strongest_features_trees.index[i] for i in range(len(strongest_features_trees))]
lasso_index_names=[strongest_features.index[i] for i in range(len(strongest_features))]
strongest_lasso_vals=[strongest_features[i] for i in strongest_features.index]
strongest_tree_vals=[strongest_features_trees[i] for i in strongest_features_trees.index]

fig, feat = plt.subplots(nrows=1, ncols=2, figsize=(30, 30))
feat[0].bar(lasso_index_names, strongest_lasso_vals)
feat[0].title.set_text("Lasso Parameters")
feat[1].bar(tree_index_names ,strongest_tree_vals)
feat[1].title.set_text("Random Forest Parameters")
plt.setp(feat[0].get_xticklabels(), rotation=90)
plt.setp(feat[1].get_xticklabels(), rotation=90)
plt.show()

# Unscaled best training and test features
strongest_predictors=features_train[strongest_features.index]
unscaled_test=features_test[strongest_features.index]

# Most important features that are standardized to be used for visualization
standardize_best_features=standardize_full[strongest_predictors.columns]
