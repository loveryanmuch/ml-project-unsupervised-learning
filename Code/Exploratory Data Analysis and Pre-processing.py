import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
wholesale_data = pd.read_csv("C:\\Users\\17789\\LHL\\ml-project-unsupervised-learning\\Wholesale_Data.csv")

# Display the first few rows
print(wholesale_data.head())

# Pairplot to visualize relationships
sns.pairplot(wholesale_data)
plt.show()

# Checking for missing values
print(wholesale_data.isnull().sum())

# Capping outliers
for column in wholesale_data.columns:
    percentiles = wholesale_data[column].quantile([0.01, 0.99]).values
    wholesale_data[column] = wholesale_data[column].clip(percentiles[0], percentiles[1])
