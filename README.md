
## Housing Price Forecast

### Description 
This project focuses on attempting to predict the Sales Prices of various houses using various regression methods on the [Kaggle Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). I will also apply clustering techniques to determine whether features can provide insight into the Sales Price of houses.

## Table of Contents
1. [Data Cleaning](#cleaning)
2. [Data processing](#processing)
3. [Feature Selection](#features)
4. [Feature Engineering](#engineering)
5. [Clustering](#clustering)
6. [Model Testing](#testing)
7. [Results](#results)


### Tools used
- Python
    - sklearn
    - pandas
    - numpy
    - matplotlib
    - scipy
    - seaborn
    - xgboost

### Methodology
<a name="cleaning"/></a>
#### Data Cleaning  

We first begin by turning our train.csv file that is given to us by Kaggle into a dataframe. 
Next, I checked the percentage of entries within a feature that were missing. This resulted in the following:

| Feature         | Percentage Missing    | 
|-----------------|-----------------------| 
| 'LotFrontage'   | 0.1773972602739726    | 
|  'Alley'        | 0.9376712328767123    | 
|  'MasVnrType'   | 0.005479452054794521  | 
|  'MasVnrArea'   | 0.005479452054794521  | 
|  'BsmtQual'     | 0.025342465753424658  | 
|  'BsmtCond'     | 0.025342465753424658  | 
|  'BsmtExposure' | 0.026027397260273973  | 
|  'BsmtFinType1' | 0.025342465753424658  | 
|  'BsmtFinType2' | 0.026027397260273973  | 
|  'Electrical'   | 0.0006849315068493151 | 
|  'FireplaceQu'  | 0.4726027397260274    | 
|  'GarageType'   | 0.05547945205479452   | 
|  'GarageYrBlt'  | 0.05547945205479452   | 
|  'GarageFinish' | 0.05547945205479452   | 
|  'GarageQual'   | 0.05547945205479452   | 
|  'GarageCond'   | 0.05547945205479452   | 
|  'PoolQC'       | 0.9952054794520548    | 
|  'Fence'        | 0.8075342465753425    | 
|  'MiscFeature'  | 0.963013698630137     | 

 
 I decided to remove the features with more than 20% of their entries missing. For all the other features, I separated the process of imputing values based upon whether the data type was categorical or numerical.
 
##### Categorical Data
For categorical data, I imputed the most frequent value when the share of an item in the feature space exceeded 66%. If this was not the case, I would randomly impute a value based upon the probability of the known values.

##### Numerical Data

For numerical data, I used the k-nearest neighbors algorithm to impute values for missing observations. First, I standardized the data and then applied the k-nearest neighbors with 10 neighbors.

<a name="processing"/></a>
#### Data Processing
After imputing in the missing values, I removed the 'Id' feature because it was not relevant to the sales price of houses. Then, I turned the 'MSubClass' numerical values into categorical values. This was done to avoid a misinterpretation of this feature when we run regression models on the data.

I decided to check the 'Lot Area' and 'Group Living Area' features for outliers because these features have the highest values out of all the features in the dataset and could be the most prone to outliers. Outliers can have a significant effect on the regression model and create less accurate predictions.  Since I had large values that were greater than the others for both of these features, I removed the 4 highest observation values from both of them. 

<a name="engineering"/></a>
#### Feature Engineering
I decided to combine pre-existing features in order to create a single feature out of them. The features that I created were 'Bathrooms' and 'Total Inside Area'. Since there are already many features, I decided it would be best to consolidate features to search for additional insight that may not be contained by the individual features.

'Bathrooms' = 'Basement Full Bathrooms' + 'Above Grade Full Bathrooms' + 'Basement Half Bathrooms' + 'Above Grade Half Bathrooms'

'Total Inside Area' = 'Square Feet of Above Grade Living Area' + 'Total Square Feet of Basement Area'

<a name="features"/></a>
#### Feature Selection
Feature selection allows me to create a more interpretable regression model by choosing important features and discarding of those that will not benefit my model. To prepare for this, I applied one-hot encoding to my categorical features. Then, I standardized all my features so that they would have a mean of 0 and variance of 1. Since my Sales Price column was large, I applied a log transformation to limit the scale of the outputs. After doing this, the data is adequate to begin feature selection.

 I applied two different methods when searching for the best features for my model. 

##### Highest Correlation
I took the correlation of all the features with the output and chose the features with the highest correlation values. I took the highest 6 features which were 'Garage Area', 'Bathrooms', 'Total Inside Area', 'Overall Quality', 'Garage Cars', and 'Square Feet of Above Group Living Area'. 

Since 'Square Feet of Above Group Living Area' is contained as a part of 'Total Inside Area' and 'Garage Cars' has a high correlation value with 'Garage Area', we will drop these 2 features. Thus, our final set of features based upon correlation are 'Garage Area', 'Bathrooms', 'Total Inside Area', and 'Overall Quality'.

##### Strongest Lasso Parameters
I checked whether my feature space would satisfy the conditions for Linear Regression by looking at whether my 'Total Inside Area' and 'Log of Sale Price' were linear along with 'Log of Sale Price' meeting the normality condition.

Since these features which are most inclined not to meet the assumptions appear to satisfy the condition of linear regression, I decided to fit a Lasso Regression model to lower the amount of parameters my model contains. Lasso utilizes the L1 norm which eliminates the less important features.

I then chose the 8 highest Lasso Regression parameters as the basis for my model.

Since we have some common features between the correlation and Lasso feature selection techniques, we will combine the feature spaces. 

<a name="clustering"/></a>
#### Clustering
An area that I decided to explore is whether we can subdivide our feature space such that we have a general idea about the range of house prices without the labels.

##### Exploring the Feature Space 
I explored the feature spaces that were generated by the correlation method of feature selection and the Lasso method. I plotted out variables which contained more than a few features as to see if there could be any predetermined clustering groups.

I selected the numerical values with many different values from the combined feature space and standardize the feature values with mean 0 and variance 1. Then, I will fit a Principal Components Analysis on the model. Then, I will plot the principal components against each other to visualize any signs of clustering between the principal components.

We will now apply two different methods of clustering and determine whether our features can provide insight into the Sales Price of houses.

##### K-Nearest Neighbors
I applied the k-nearest neighbors clustering algorithm to my first and third principal component that I obtained from principal component analysis.  I have decided to use 6 clusters because that is the point where the decreases in the sum of square error do not change.

As a result, I have the following cluster below.

I will now compare this to the actual data with 6 clusters. 

There exists a similarity between the clusters in the pictures. We can divide housing prices by price groups by applying k-nearest neighbors on principal components.

##### Hierarchical Clustering
I used the standardized features for my hierarchical clustering scheme. This produced the dendrogram below. Based upon the dendrogram, a good amount of clusters to segment the data into will be 7 clusters.

I decided to use agglomerative clustering with ward linkage to predict the labels of my features.

 I proceeded to plot graphs of features where the true housing price was segmented into 6 groups. 

The clusters produced from agglomerative clustering are similar to those of the true sales price.

<a name="testing"/></a>
#### Model Testing
After selecting the features through feature selection techniques, I have decided to to test three different set of inputs on my regression models. The inputs that I will be testing are the  features selected by correlation, the features selected by Lasso, and the combined feature space of both methods.

I will split my data into training and test data at an 80% train and 20% test split.

I will train the Random Forest Regressor Model, Support Vector Regressor Model, K-Nearest Neighbors regressor model, XGBoost model, and Adaboost model with my inputs. 

I will evaluate each of these models by doing a Random Search on the parameters with my training data. Then, I will do a 10-fold cross-validation to evaluate the mean absolute error for each of my regression models.

Lastly, I will fit my models with the training data and predict the Sales Prices of the test data in terms of mean squared error.

<a name="results"/></a>
#### Results
