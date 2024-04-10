import pandas as pd
import numpy as np

# Load the dataset from CSV
df = pd.read_csv('dataset.csv')

# Handling missing values
df = df.dropna()  # Drop rows with any missing values
# Alternative: df = df.fillna(0) to fill missing values with zeros

# Removing duplicates
df = df.drop_duplicates()

# Handling outliers (assuming a numerical column named 'value')
z_scores = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
df = df[z_scores < 3]  # Keep rows within 3 standard deviations

# Data normalization (assuming a numerical column named 'value')
df['value'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())

# Feature scaling (assuming numerical columns named 'feature1' and 'feature2')
df['feature1'] = (df['feature1'] - df['feature1'].mean()) / df['feature1'].std()
df['feature2'] = (df['feature2'] - df['feature2'].mean()) / df['feature2'].std()

# Data encoding (assuming a categorical column named 'category')
df = pd.get_dummies(df, columns=['category'])

# Date/time conversion (assuming a column named 'timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_dataset.csv', index=False)

# Load the dataset from CSV
df = pd.read_csv('dataset.csv')

# Dropping irrelevant columns
df = df.drop(['column1', 'column2'], axis=1)

# Renaming columns
df = df.rename(columns={'old_column_name': 'new_column_name'})

# Handling missing values
df['column3'] = df['column3'].fillna(df['column3'].mean())
df['column4'] = df['column4'].fillna('Unknown')

# Removing leading/trailing whitespaces
df['column5'] = df['column5'].str.strip()

# Converting data types
df['column6'] = pd.to_numeric(df['column6'], errors='coerce')
df['column7'] = pd.to_datetime(df['column7'])

# Handling categorical variables
df['column8'] = df['column8'].astype('category')

# Binning numerical data
df['column9'] = pd.cut(df['column9'], bins=[0, 10, 20, 30, 40], labels=['A', 'B', 'C', 'D'])

# Handling outliers (assuming a numerical column named 'column10')
q1 = df['column10'].quantile(0.25)
q3 = df['column10'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['column10'] >= lower_bound) & (df['column10'] <= upper_bound)]

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_dataset.csv', index=False)