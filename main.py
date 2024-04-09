import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
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


import pandas as pd

# Read the CSV file into a DataFrame

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
plt.show()

# Get the 'id' column and count unique values
import pandas as pd

# Assuming you have already loaded your DataFrame into dfs['attributes']


# Correcting the key used to access the DataFrame from the dictionary
brand_column = dfs['attributes']['name']

# Get the five most occurring brands and their frequencies
top_brands = brand_column.value_counts().head(5)

print("The five most occurring brands in the attributes data and their frequencies are:")
for brand, frequency in top_brands.items():
    print("Brand:", brand, "| Frequency:", frequency)
