import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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

from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Assuming dfs['train.csv'] is your DataFrame
# Features (X) and Labels (y)
import pandas as pd

# Load the attributes data
attributes = pd.read_csv('attributes.csv')
# Before combining 'name' and 'value', convert them to strings to ensure compatibility
attributes['combined'] =  attributes['value'].astype(str)

attributes_combined = attributes.groupby('product_uid')['combined'].apply(lambda x: ' | '.join(x)).reset_index()
# Merge the main dataset with the combined attributes
merged_data = pd.merge(dfs['train.csv'], attributes_combined, on='product_uid', how='left')
print(merged_data.head(5))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np

# Define your feature and target variables
X = merged_data[['combined','id']]  # 'combined' is your new aggregated attribute feature
y = merged_data['relevance']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# You might need to fill NaN values for text processing
X_train['combined'] = X_train['combined'].fillna('')
X_test['combined'] = X_test['combined'].fillna('')

# Example: Using TfidfVectorizer for the 'combined' column and a RandomForestRegressor for prediction
tfidf = TfidfVectorizer(stop_words='english', max_features=2)  # Adjust parameters as needed

# Example: Assuming a way to apply TfidfVectorizer to 'combined' and using existing features
# In practice, you'll need to properly combine these features, potentially using FeatureUnion or a custom transformer

# Fitting the TF-IDF on the combined column (for simplicity, this example does not combine with other features)
X_train_tfidf = tfidf.fit_transform(X_train['combined'])
X_test_tfidf = tfidf.transform(X_test['combined'])

# Training the RandomForestRegressor (consider using a pipeline for a more elegant solution)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_tfidf, y_train)

# Making predictions and evaluating
y_pred = model.predict(X_test_tfidf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

