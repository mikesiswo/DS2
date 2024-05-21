import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np

# File paths for CSV files
file_paths = ['test.csv', 'train.csv', 'product_descriptions.csv', 'attributes.csv']

# Dictionary to store DataFrames
dfs = {}

# Read CSV files into DataFrames
for file_path in file_paths:
    dfs[file_path] = pd.read_csv(file_path, encoding="MacRoman" if 'csv' in file_path else None)

# Number of unique rows in 'id' column
unique_values_count = len(dfs['train.csv']['id'].unique())
print("Number of rows without duplicates in the 'id' column:", unique_values_count)

# Number of unique rows in 'product_uid' column
unique_values_count_product = len(dfs['train.csv']['product_uid'].unique())
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

# Descriptive statistics for the 'relevance' column
relevance_stats = dfs['train.csv']['relevance'].describe()
print("Descriptive statistics for the 'relevance' column:")
print(relevance_stats)

# Histogram of the distribution of relevance values
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

# Combine 'name' and 'value' columns and merge with the main dataset
attributes = dfs['attributes.csv']
attributes['combined'] = attributes['value'].astype(str)
attributes_combined = attributes.groupby('product_uid')['combined'].apply(lambda x: ' | '.join(x)).reset_index()
merged_data = pd.merge(dfs['train.csv'], attributes_combined, on='product_uid', how='left')

# Features (X) and Labels (y)
X = merged_data[['combined', 'id']]
y = merged_data['relevance']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill NaN values for text processing
X_train['combined'] = X_train['combined'].fillna('')
X_test['combined'] = X_test['combined'].fillna('')

# TfidfVectorizer for text processing
tfidf = TfidfVectorizer(stop_words='english', max_features=2)
X_train_tfidf = tfidf.fit_transform(X_train['combined'])
X_test_tfidf = tfidf.transform(X_test['combined'])

# RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_tfidf, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test_tfidf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")
