from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
import matplotlib.pyplot as plt
import pickle
from housingprediction import best_train_features, best_test_features, y_train, y_test
import time

# Test with the Random Forest Model
best_rf_params=RandomForestRegressor()
rf_trees=[i*10 for i in range(1,21)]
rf_params={'n_estimators':rf_trees}

best_rf=RandomizedSearchCV(best_rf_params, rf_params, n_iter=15, n_jobs=-1, cv=5).fit(best_train_features, y_train)
best_rf_model=RandomForestRegressor(**best_rf.best_params_)
# Check performance of the Random Forest model

best_rf_model=best_rf_model.fit(best_train_features, y_train)
rf_pred=best_rf_model.predict(best_test_features)
pickle.dump(best_rf_model, open("rf_model.sav", "wb"))

# Test with XGBoost Model
best_xgb_params=XGBRegressor()
start_time=time.time()
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

best_xgb=RandomizedSearchCV(best_xgb_params, random_grid, n_iter=15, n_jobs=-1, cv=5).fit(best_train_features, y_train)
best_xgb_model=XGBRegressor(**best_xgb.best_params_)
best_xgb_model=best_xgb_model.fit(best_train_features, y_train)
xgb_pred=best_xgb_model.predict(best_test_features)
pickle.dump(best_xgb_model, open("xgb_model.sav", "wb"))

print("Time taken is:", time.time()-start_time, " s")

# Test with AdaBoost Model
best_ada_params=AdaBoostRegressor()
start_time=time.time()

n_estimators=[10,50,100,200]
learning_rate=[0.01, 0.1, 0.5, 1, 2]
loss=["linear", "square", "exponential"]
base_estimator=[DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=7), DecisionTreeRegressor(max_depth=10)]

random_grid={'n_estimators':n_estimators,
             'learning_rate':learning_rate,
             'loss':loss, 'base_estimator':base_estimator}

best_ada=RandomizedSearchCV(best_ada_params, random_grid, n_iter=15, n_jobs=-1, cv=5).fit(best_train_features, y_train)
best_ada_model=AdaBoostRegressor(**best_ada.best_params_)
best_ada_model=best_ada_model.fit(best_train_features, y_train)
ada_pred=best_ada_model.predict(best_test_features)
pickle.dump(best_ada_model, open("ada_model.sav", "wb"))

print("Time taken is:", time.time()-start_time, " s")

# Compare the errors in the models, predictions in models
rf_mse=mean_squared_error(y_test, rf_pred)
rf_mae=mean_absolute_error(y_test, rf_pred)
xgb_mse=mean_squared_error(y_test, xgb_pred)
xgb_mae=mean_absolute_error(y_test, xgb_pred)
ada_mse=mean_squared_error(y_test, ada_pred)
ada_mae=mean_absolute_error(y_test, ada_pred)
test_amts=[i for i in range(len(y_test))]

models=["Random Forest", "XGBoost", "Adaboost"]
results=[rf_mse, xgb_mse, ada_mse]

plt.figure(figsize=(15,15))
plt.plot(models, results)
plt.show()

plt.figure(figsize=(50,15))
plt.title("Adaboost vs. Actual")
plt.plot(test_amts, y_test, color="red")
plt.plot(test_amts, ada_pred, color="blue")
plt.show()

plt.figure(figsize=(50,15))
plt.title("XGBoost vs. Actual")
plt.plot(test_amts, y_test, color="red")
plt.plot(test_amts, xgb_pred, color="black")
plt.show()



