import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from scipy.stats import uniform

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

# Count unique values in the 'id' and 'product_uid' columns
id_column = dfs['train.csv']['id']
unique_values_count = len(id_column.unique())
print("Number of rows without duplicates in the 'id' column:", unique_values_count)

product_column = dfs['train.csv']['product_uid']
unique_values_count_product = len(product_column.unique())
print("Number of rows without duplicates in the 'product' column:", unique_values_count_product)

# Get the 2 most occurring products
top_product_uids = dfs['train.csv']['product_uid'].value_counts().head(2)

# Preprocess the attributes csv to avoid NaN values
dfs['attributes'].dropna(inplace=True)

for product_uid, frequency in top_product_uids.items():
    # Filter rows in attributes where 'name' contains the current product_uid
    product_attributes = dfs['attributes'][dfs['attributes']['product_uid'].astype(str).str.contains(str(product_uid))]
    
    # Filter rows where 'name' contains 'MFG Brand Name'
    brands = product_attributes['name'].str.contains("MFG Brand Name")
    
    # Get corresponding 'value' column (product names)
    product_name = product_attributes[brands]['value']
    
    # Print the product_uid and its corresponding product name
    print("Product UID:", product_uid)
    print("Product Name:", product_name.values)
    print("Frequency:", frequency)
    
# Get descriptive statistics for the 'relevance' column
relevance_stats = dfs['train.csv']['relevance'].describe()
mean_relevance = relevance_stats['mean']
median_relevance = relevance_stats['50%']  # Median
std_relevance = relevance_stats['std']
print("Descriptive statistics for the 'relevance' column:")
print("Mean:", mean_relevance)
print("Median:", median_relevance)
print("Standard Deviation:", std_relevance)

# Plot distribution of relevance values
plt.figure(figsize=(10, 6))
sns.histplot(data=dfs['train.csv'], x='relevance', bins=20, kde=True)
plt.title('Distribution of Relevance Values')
plt.xlabel('Relevance')
plt.ylabel('Frequency')
plt.savefig('histogram.png')

# Get the five most occurring brands in the attributes data and their frequencies
# Filter rows where 'name' contains 'MGF Brand Name'
brands = dfs['attributes']['name'].str.contains("MFG Brand Name")
# Get corresponding 'value' column
brand_names = dfs['attributes'][brands]['value']
top_brands = brand_names.value_counts().head(5)
print("The five most occurring brands in the attributes data and their frequencies are:")
for brand, frequency in top_brands.items():
    print("Brand:", brand, "| Frequency:", frequency)

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
X_train = merged_data[['search_term','combined','product_title','product_description']]

# Labels (y)
y_train = merged_data['relevance']

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and fit the TargetEncoder
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train_split, y_train_split)
X_test_encoded = encoder.transform(X_test_split)

# Set up the parameter grid for Ridge regression
param_dist = {'alpha': uniform(0.1, 10)}

# Create a Ridge regression model instance
ridge_model = Ridge()

# Perform randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=ridge_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=1)
random_search.fit(X_train_encoded,y_train_split)

# Print the best parameters and the corresponding RMSE
best_model = random_search.best_estimator_
print("Best parameters found: ", random_search.best_params_)
best_rmse = np.sqrt(-random_search.best_score_)
print("Best RMSE: ", best_rmse)

# Capture the end time
end_time = time.time()
# Calculate the processing time
processing_time = end_time - start_time
# Print the processing time
print("Processing time:", processing_time, "seconds")


#----------------------------
# RIDGE REGRESSION 
# model_ridge_regression = Ridge(alpha=1.0) 
# model_ridge_regression.fit(X_train_encoded, y_train_split)
# y_pred_ridge_regression = model_ridge_regression.predict(X_test_encoded)
# rmse_ridge_regression = np.sqrt(mean_squared_error(y_test_split, y_pred_ridge_regression))
# print(f"Root Mean Squared Error (RMSE) using RidgeRegression and TargetEncoder: {rmse_ridge_regression}")
#----------------------------
# LINEAR REGRESSION
# model_linear_regression = LinearRegression()
# model_linear_regression.fit(X_train_encoded, y_train_split)
# y_pred_linear_regression = model_linear_regression.predict(X_test_encoded)
# rmse_linear_regression = np.sqrt(mean_squared_error(y_test_split, y_pred_linear_regression))
# print(f"Root Mean Squared Error (RMSE) using LinearRegression and TargetEncoder: {rmse_linear_regression}")
#----------------------------
# RANDOMFOREST REGRESSOR
# model_target_encoder = RandomForestRegressor(random_state=42)
# model_target_encoder.fit(X_train_encoded, y_train_split)
# y_pred_target_encoder = model_target_encoder.predict(X_test_encoded)
# rmse_target_encoder = np.sqrt(mean_squared_error(y_test_split, y_pred_target_encoder))
# print(f"Root Mean Squared Error (RMSE) using Random Forest and TargetEncoder: {rmse_target_encoder}")
#----------------------------
# DECISSIONTREEREGRESSOR     
# model_decision_tree = DecisionTreeRegressor(random_state=42)
# model_decision_tree.fit(X_train_encoded, y_train_split)
# y_pred_decision_tree = model_decision_tree.predict(X_test_encoded)
# rmse_decision_tree = np.sqrt(mean_squared_error(y_test_split, y_pred_decision_tree))
# print(f"Root Mean Squared Error (RMSE) using DecissionTreeRegressor and TargetEncoder: {rmse_decision_tree}")
#----------------------------