import pandas as pd

# Load dataset
data = pd.read_csv("D:\\Downloads\\winequality-red.csv", sep=';')

# Check null values
print("Null values in dataset:")
print(data.isnull().sum())

# Remove null values
data_clean = data.dropna()

print("Dataset shape after cleaning:", data_clean.shape)