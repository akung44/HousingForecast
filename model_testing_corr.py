from xgboost import XGBRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
from housing_preprocessing_raw import X_train, y_train, X_test, y_test, corr_index_names
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

random.seed(42)
# Standardize input and outputs for certain models
best=X_train[corr_index_names]
scaled_x_fit=StandardScaler().fit(best)
scaled_x=scaled_x_fit.transform(best)

# Check performance of the Random Forest model with scaled x and log transformed y
rf_model=RandomForestRegressor()
cv_rf=cross_val_score(rf_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , cv_rf)
print("The average error is", cv_rf.mean())

# Hyperparameter tuning Random Forest
rf_trees=[i*10 for i in range(1,21)]
min_leaf_samples=[i*2 for i in range(1,10)]
max_depth=[i*2 for i in range(1,25)]
rf_params={'n_estimators':rf_trees,
           'min_samples_leaf':min_leaf_samples,
           'max_depth':max_depth}

best_rf=RandomizedSearchCV(rf_model, rf_params, n_iter=25, n_jobs=-1, cv=10, scoring='neg_mean_squared_error').fit(scaled_x, y_train)
# Best Model according to Random Search
best_rf_model=RandomForestRegressor(**best_rf.best_params_)

# Check performance of Random Search model with 10-fold CV
best_cv_rf=cross_val_score(best_rf_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are", best_cv_rf)
print("The average error is", best_cv_rf.mean())

# Check performance of the Support Vector model with scaled x and log transformed y
svr_model=svm.SVR()
cv_svr=cross_val_score(svr_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , cv_svr)
print("The average error is", cv_svr.mean())

# Hyperparameter tuning Support Vector
kern=['linear', 'poly', 'rbf', 'sigmoid']
C=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
epsilon=[0.01, 0.1, 0.25, 0.5]
svr_params={'kernel':kern, 'C':C, 'epsilon': epsilon}

# Best Model According to Random Search
best_svr=RandomizedSearchCV(svr_model, svr_params, n_iter=25, n_jobs=-1, cv=10, scoring='neg_mean_squared_error').fit(scaled_x, y_train)
best_svr_model=svm.SVR(**best_svr.best_params_)

# Check performance of Random Search model with 10-fold CV
best_cv_svr=cross_val_score(best_svr_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are", best_cv_svr)
print("The average error is", best_cv_svr.mean())

# KNearestNeighbors Regressor
knn_model=KNeighborsRegressor()
cv_knn=cross_val_score(knn_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , cv_knn)
print("The average error is", cv_knn.mean())

# Hyperparameter tuning KNN
neighbors=[3,5,10,20]
weights=['uniform', 'distance']
p=[1,2]
knn_params={'n_neighbors':neighbors, 'weights':weights, 'p':p}

best_knn=RandomizedSearchCV(knn_model, knn_params, n_iter=25, n_jobs=-1, cv=10, scoring='neg_mean_squared_error').fit(scaled_x, y_train)
best_knn_model=KNeighborsRegressor(**best_knn.best_params_)

# Validate performance of Random Search Model with 10-fold CV
best_cv_knn=cross_val_score(best_knn_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are", best_cv_knn)
print("The average error is", best_cv_knn.mean())

# Original XGB model with 10-fold CV
xgb_regressor=XGBRegressor()
cv_xgb_df=cross_val_score(xgb_regressor, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , cv_xgb_df)
print("The average error is", cv_xgb_df.mean())

# Hyperparameter tuning with random search
booster=["gbtree", "gblinear", "dart"]
gamma=[0,0.01,0.1,1]
max_delta_step=[0, 0.01, 0.1, 0.5, 1]
reg_lambda=[0, 0.1, 0.5, 1]
alpha=[0, 0.1, 0.5, 1]
n_estimators=[10,50,100,200,500]

random_grid={'booster': booster, 'gamma':gamma,
        'max_delta_step':max_delta_step,
        'reg_lambda':reg_lambda,
        'alpha':alpha, 'n_estimators':n_estimators}

best_xgb=RandomizedSearchCV(xgb_regressor, random_grid, n_iter=25, n_jobs=-1, cv=10).fit(scaled_x, y_train)
best_xgb_model=XGBRegressor(**best_xgb.best_params_)

# 10-fold cross validation for the new XGB model
best_xgb=cross_val_score(best_xgb_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , best_xgb)
print("The average error is", best_xgb.mean())

# 10-fold cross validation for the AdaBoost model
ada_model=AdaBoostRegressor()
cv_ada=cross_val_score(ada_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , cv_ada)
print("The average error is", cv_ada.mean())

# Random Search for best parameters
n_estimators=[10,50,100,200]
learning_rate=[0.01, 0.1, 0.5, 1, 2]
loss=["linear", "square", "exponential"]
base_estimator=[DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=7), DecisionTreeRegressor(max_depth=10)]

random_grid={'n_estimators':n_estimators,
             'learning_rate':learning_rate,
             'loss':loss, 'base_estimator':base_estimator}

best_ada=RandomizedSearchCV(ada_model, random_grid, n_iter=25, n_jobs=-1, cv=5).fit(scaled_x, y_train)
best_ada_model=AdaBoostRegressor(**best_ada.best_params_)

# 10-fold cross validation for the new AdalBoost  model
best_ada_cv=cross_val_score(best_ada_model, scaled_x, y_train, cv=10, scoring='neg_mean_absolute_error')
print("The errors for each fold are" , best_ada_cv)
print("The average error is", best_ada_cv.mean())


# Fitting the entire dataset
full_x_train=StandardScaler().fit(X_train)
standardized_full_train=full_x_train.transform(X_train)
standardized_full_test=full_x_train.transform(X_test)

# Fitting models with training data
best_rf_model.fit(standardized_full_train, y_train)
best_xgb_model.fit(standardized_full_train, y_train)
best_ada_model.fit(standardized_full_train, y_train)
best_svr_model.fit(standardized_full_train, y_train)
best_knn_model.fit(standardized_full_train, y_train)

rf_full=best_rf_model.predict(standardized_full_test)
xgb_full=best_xgb_model.predict(standardized_full_test)
ada_full=best_ada_model.predict(standardized_full_test)
svr_full=best_svr_model.predict(standardized_full_test)
knn_full=best_knn_model.predict(standardized_full_test)

# Fitting on full features predictions
rf_mse_full=mean_squared_error(y_test, np.exp(rf_full))
rf_mae_full=mean_absolute_error(y_test, np.exp(rf_full))
xgb_mse_full=mean_squared_error(y_test, np.exp(xgb_full))
xgb_mae_full=mean_absolute_error(y_test, np.exp(xgb_full))
ada_mse_full=mean_squared_error(y_test, np.exp(ada_full))
ada_mae_full=mean_absolute_error(y_test, np.exp(ada_full))
svr_mse_full=mean_squared_error(y_test, np.exp(svr_full))
svr_mae_full=mean_absolute_error(y_test, np.exp(svr_full))
knn_mse_full=mean_squared_error(y_test, np.exp(knn_full))
knn_mae_full=mean_absolute_error(y_test, np.exp(knn_full))

# Plotting the models
test_amts=[i for i in range(len(y_test))]

models=["Random Forest", "XGBoost", "Adaboost", "SVR", "KNN"]
results=[rf_mse_full, xgb_mse_full, ada_mse_full, svr_mse_full, knn_mse_full]

for i in range(5):
    print(models[i], results[i])

plt.figure(figsize=(15,15))
plt.plot(models, results)
plt.show()
