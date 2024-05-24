import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import re
import time
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from scipy.stats import uniform
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
from scipy.stats import uniform
from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor

# Mapping dictionary for unit normalization
unit_mapping = {
    ' ft': ' ft.',
    ' feet': ' ft.',
    ' pound': ' lbs.',
    ' lbs': ' lbs.',
    ' in': ' in.',
    ' inch': ' in.',
    ' inches': ' in.',
    ' yd': 'yards',
    'mdf': 'MDF',
    'oz': 'ounces',
    ' lb': 'pound',
    ' pounds': 'pounds',
    ' pound': 'pounds',
    ' gal': 'gallons',
    ' pint': ' pt.',
    'sq': ' sq.',
    'cubic': ' cu.',
    ' cu': ' cu.',
}

def preprocess_text(text):
    # Split the text into words
    words = text.split()
    
    # Replace words based on the unit_mapping dictionary
    for i, word in enumerate(words):
        if word.lower() in unit_mapping:
            words[i] = unit_mapping[word.lower()]
    
    # Join the words back into a single string
    return ' '.join(words)

# Capture the start time
start_time = time.time()

# Dictionary to store DataFrames
dfs = {}

# Read datasets into DataFrames
dfs['test.csv'] = pd.read_csv('test.csv', encoding="MacRoman")
dfs['train.csv'] = pd.read_csv('train.csv', encoding="MacRoman")
dfs['product_descriptions.csv'] = pd.read_csv('product_descriptions.csv', encoding="ascii")
dfs['attributes'] = pd.read_csv('attributes.csv', encoding="utf-8")

print(dfs['attributes'].head(5))

# Preprocess text in the 'search_term' column
df_train = dfs['train.csv']
df_train["search_term"] = df_train["search_term"].apply(preprocess_text)

# Merge attributes data
attributes = pd.read_csv('attributes.csv')
attributes['combined'] =  attributes['value'].astype(str)
attributes_combined = attributes.groupby('product_uid')['combined'].apply(lambda x: ' | '.join(x)).reset_index()
merged_data = pd.merge(df_train, attributes_combined, on='product_uid', how='left')

# Merge product descriptions
descriptions = pd.read_csv('product_descriptions.csv')
descriptions_uid = descriptions.groupby('product_uid')['product_description'].first().reset_index()
merged_data = pd.merge(merged_data, descriptions_uid, on='product_uid', how='left')

# Features (X)
X_train = merged_data[['search_term','combined','product_title','product_description','product_uid']]

# Labels (y)
y_train = merged_data['relevance']

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and fit the TargetEncoder
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train_split, y_train_split)
X_test_encoded = encoder.transform(X_test_split)


# Define the base models
base_models = [
    ('xgboost', xgb.XGBRegressor()),
    ('lightgbm', lgb.LGBMRegressor())
]

# # Create the Stacking Regressor with base models and final estimator
# stacking_model = StackingRegressor(
#     estimators=base_models,
#     final_estimator=Ridge()
# )
#---------------------------------------------------------------------------------------------------------
# STACKING MODELS BASE - META :
# Processing time: 80.89748001098633 seconds
# Final RMSE on test set:  0.5328690801039294
# ----------------------------
# from scipy.stats import randint, uniform
# # Perform hyperparameter tuning for the stacking model
# param_dist_stacking = {
#     'final_estimator__alpha': uniform(0.1, 10)
# }

# random_search_stacking = RandomizedSearchCV(estimator=stacking_model, param_distributions=param_dist_stacking, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=1)
# random_search_stacking.fit(X_train_encoded, y_train_split)

# # Make predictions on the test set
# y_pred = random_search_stacking.best_estimator_.predict(X_test_encoded)
#---------------------------------------------------------------------------------------------------------
# VOTING REGRESSOR : 
# RMSE: 0.5346767043946465
# Total time :  4.98853611946106
# ----------------------------
# from sklearn.ensemble import VotingRegressor
# # Create the Voting Regressor with base models
# voting_model = VotingRegressor(estimators=base_models)

# # Fit the voting model
# voting_model.fit(X_train_encoded, y_train_split)

# # Make predictions on the test set
# y_pred = voting_model.predict(X_test_encoded)
#---------------------------------------------------------------------------------------------------------
# Model Selection 
# RMSE: 0.5322716923040763
# Total time :  18.83643889427185
# ----------------------------
# from scipy.stats import randint
# # Define the models and their hyperparameter distributions
# models = {
#     'xgboost': {
#         'model': xgb.XGBRegressor(),
#         'params': {
#             'n_estimators': randint(50, 200),
#             'max_depth': randint(3, 10),
#             'learning_rate': uniform(0.01, 0.3)
#         }
#     },
#     'lightgbm': {
#         'model': lgb.LGBMRegressor(),
#         'params': {
#             'n_estimators': randint(50, 200),
#             'max_depth': randint(3, 10),
#             'learning_rate': uniform(0.01, 0.3)
#         }
#     }
# }

# best_model = None
# best_score = float('inf')
# best_model_name = ""

# for name, model_info in models.items():
#     print(f"Training {name} model...")
#     random_search = RandomizedSearchCV(estimator=model_info['model'], param_distributions=model_info['params'], n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=1)
#     random_search.fit(X_train_encoded, y_train_split)
    
#     # Compare performance
#     score = -random_search.best_score_
#     print(f"Best score for {name}: {score}")
    
#     if score < best_score:
#         best_score = score
#         best_model = random_search.best_estimator_
#         best_model_name = name

# print(f"Best model: {best_model_name} with score: {best_score}")

# # Make predictions on the test set
# y_pred = best_model.predict(X_test_encoded)
#---------------------------------------------------------------------------------------------------------
# # Random SubSpace Method 
# RMSE: 0.5462755234541994
# Total time :  661.3004431724548
# ----------------------------
# # Define the base models with Random Subspace Method
# base_models = [
#     ('xgboost', BaggingRegressor(estimator=xgb.XGBRegressor(), n_estimators=10, max_features=0.8, random_state=42)),
#     ('lightgbm', BaggingRegressor(estimator=lgb.LGBMRegressor(), n_estimators=10, max_features=0.8, random_state=42))
# ]

# # Create the Bagging Regressor with base models
# random_subspace_model = StackingRegressor(
#     estimators=base_models,
#     final_estimator=Ridge()
# )

# # Perform hyperparameter tuning for the final estimator in the stacking model
# param_dist_stacking = {
#     'final_estimator__alpha': uniform(0.1, 10)
# }

# random_search_subspace = RandomizedSearchCV(estimator=random_subspace_model, param_distributions=param_dist_stacking, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=1)
# random_search_subspace.fit(X_train_encoded, y_train_split)

# # Make predictions on the test set
# y_pred = random_search_subspace.best_estimator_.predict(X_test_encoded)
#---------------------------------------------------------------------------------------------------------
# Average Prediction Method 
# RMSE: 0.5346767043946465
# Total time :  5.000196933746338
# ----------------------------
# # Train base models and collect predictions
# predictions = []

# for name, model in base_models:
#     model.fit(X_train_encoded, y_train_split)
#     pred = model.predict(X_test_encoded)
#     predictions.append(pred)

# # Average the predictions
# average_pred = np.mean(predictions, axis=0)

# # Calculate RMSE
# rmse = np.sqrt(mean_squared_error(y_test_split, average_pred))
# print(f'RMSE: {rmse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
print(f'RMSE: {rmse}')
# Capture the start time
end_time = time.time()
print("Total time : ",end_time-start_time)





# # Print the best parameters and the corresponding RMSE for Stacking Regressor
# best_stacking_model = random_search_stacking.best_estimator_
# print("Best parameters for Stacking Regressor: ", random_search_stacking.best_params_)
# best_rmse_stacking = np.sqrt(-random_search_stacking.best_score_)
# print("Best RMSE for Stacking Regressor: ", best_rmse_stacking)

# # Capture the end time
# end_time = time.time()
# # Calculate the processing time
# processing_time = end_time - start_time
# # Print the processing time
# print("Processing time:", processing_time, "seconds") 

# print("----------------------------")

# # Evaluate the final model on the test set to get the RMSE
# y_pred = best_stacking_model.predict(X_test_encoded)
# final_rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))
# print("Final RMSE on test set: ", final_rmse)


#--------------------------------------------------------------------------------
# # Define the base models
# base_models = [
#     ('random_forest', RandomForestRegressor()),
#     ('gradient_boosting', GradientBoostingRegressor()),
#     ('svr', SVR())
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 1.6601864044243653}
# Best RMSE for Stacking Regressor:  0.2118441600677186
# Processing time: 7852.365780830383 seconds
# ----------------------------
# Final RMSE on test set:  0.5350444971465439
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('gradient_boosting', GradientBoostingRegressor()),
#     ('decision_tree', DecisionTreeRegressor()),
#     ('knn', KNeighborsRegressor())
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 6.086584841970366}
# Best RMSE for Stacking Regressor:  0.21307513592176028
# Processing time: 410.8912229537964 seconds
# ----------------------------
# Final RMSE on test set:  0.5339486288197459
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('gradient_boosting', GradientBoostingRegressor()),
#     ('decision_tree', DecisionTreeRegressor())
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 3.845401188473625}
# Best RMSE for Stacking Regressor:  0.2151985185205484
# Processing time: 409.19667196273804 seconds
# ----------------------------
# Final RMSE on test set:  0.5342835313273167
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('adaboost', AdaBoostRegressor()),
#     ('lightgbm', lgb.LGBMRegressor()),
#     ('extra_trees', ExtraTreesRegressor())
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 1.6601864044243653}
# Best RMSE for Stacking Regressor:  0.20597413879699025
# Processing time: 471.10960054397583 seconds
# ----------------------------
# Final RMSE on test set:  0.5333852597320776
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('lasso', Lasso()),
#     ('bagging', BaggingRegressor()),
#     ('catboost', CatBoostRegressor(silent=True))  # CatBoost with silent mode on to suppress output
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 9.60714306409916}
# Best RMSE for Stacking Regressor:  0.20622234099347353
# Processing time: 451.0847578048706 seconds
# ----------------------------
# Final RMSE on test set:  0.5356618738988406
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('xgboost', xgb.XGBRegressor()),
#     ('lightgbm', lgb.LGBMRegressor()),
#     ('catboost', CatBoostRegressor(silent=True))  # CatBoost with silent mode on to suppress output
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 9.60714306409916}
# Best RMSE for Stacking Regressor:  0.20546866687187765
# Processing time: 392.94754695892334 seconds
# ----------------------------
# Final RMSE on test set:  0.5334314447324029
#--------------------------------------------------------------------------------
# Define the base models
# base_models = [
#     ('xgboost', xgb.XGBRegressor()),
#     ('lightgbm', lgb.LGBMRegressor())  # CatBoost with silent mode on to suppress output
# ]
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Best parameters for Stacking Regressor:  {'final_estimator__alpha': 6.086584841970366}
# Best RMSE for Stacking Regressor:  0.20598745088846027
# Processing time: 80.89748001098633 seconds
# ----------------------------
# Final RMSE on test set:  0.5328690801039294
#--------------------------------------------------------------------------------
