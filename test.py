import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from category_encoders import TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# List of file paths for your CSV files
file_paths = ['test.csv', 'train.csv', 'product_descriptions.csv', 'attributes.csv']  # Add all your file paths here

# Dictionary to store DataFrames
dfs = {}

dfs['test.csv']= pd.read_csv('test.csv', encoding="MacRoman")
dfs['train.csv']= pd.read_csv('train.csv', encoding="MacRoman")
dfs['product_descriptions.csv']= pd.read_csv('product_descriptions.csv', encoding="ascii")
dfs['attributes']= pd.read_csv('attributes.csv', encoding="utf-8")

id_column = dfs['train.csv']['id']
unique_values_count = len(id_column.unique())

print("Number of rows without duplicates in the 'id' column:", unique_values_count)

product_column = dfs['train.csv']['product_uid']
unique_values_count_product = len(product_column.unique())

print("Number of rows without duplicates in the 'product' column:", unique_values_count_product)

dfs['train.csv'] = pd.read_csv('train.csv', encoding="MacRoman")

# Get the 'id' column and count unique values
id_column = dfs['train.csv']['id']

# Get the two most occurring products and their frequencies
top_products = product_column.value_counts().head(2)

print("The two most occurring products in the training data and their frequencies are:")
for product, frequency in top_products.items():
    print("Product:", product, "| Frequency:", frequency)

# Get descriptive statistics for the 'relevance' column
relevance_stats = dfs['train.csv']['relevance'].describe()

# Extract mean, median, and standard deviation
mean_relevance = relevance_stats['mean']
median_relevance = relevance_stats['50%']  # Median
std_relevance = relevance_stats['std']

print("Descriptive statistics for the 'relevance' column:")
print("Mean:", mean_relevance)
print("Median:", median_relevance)
print("Standard Deviation:", std_relevance)


plt.figure(figsize=(10, 6))
sns.histplot(data=dfs['train.csv'], x='relevance', bins=20, kde=True)
plt.title('Distribution of Relevance Values')
plt.xlabel('Relevance')
plt.ylabel('Frequency')
plt.savefig('histogram.png')

# Correcting the key used to access the DataFrame from the dictionary
brand_column = dfs['attributes']['name']

# Get the five most occurring brands and their frequencies
top_brands = brand_column.value_counts().head(5)

print("The five most occurring brands in the attributes data and their frequencies are:")
for brand, frequency in top_brands.items():
    print("Brand:", brand, "| Frequency:", frequency)


df_train = dfs['train.csv']
df_test = dfs['test.csv']

attributes = pd.read_csv('attributes.csv')
# Before combining 'name' and 'value', convert them to strings to ensure compatibility
attributes['combined'] =  attributes['value'].astype(str)

attributes_combined = attributes.groupby('product_uid')['combined'].apply(lambda x: ' | '.join(x)).reset_index()
# Merge the main dataset with the combined attributes
merged_data = pd.merge(df_train, attributes_combined, on='product_uid', how='left')

# Features (X)
X_train = merged_data[['search_term', 'combined','product_title']]
X_test = merged_data[['search_term', 'combined','product_title']]  

# Labels (y)
y_train = merged_data['relevance']

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and fit the TargetEncoder
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train_split, y_train_split)
X_test_encoded = encoder.transform(X_test_split)

# Training the model with encoded features
model_target_encoder = RandomForestRegressor(random_state=42)
model_target_encoder.fit(X_train_encoded, y_train_split)

# Making predictions with the model using TargetEncoder
y_pred_target_encoder = model_target_encoder.predict(X_test_encoded)
rmse_target_encoder = np.sqrt(mean_squared_error(y_test_split, y_pred_target_encoder))

print(f"Root Mean Squared Error (RMSE) using Random Forest and TargetEncoder: {rmse_target_encoder}")

