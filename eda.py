import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Training.csv")

print("Original Shape:", df.shape)

# Remove empty column if present
df = df.drop(columns=["Unnamed: 133"], errors="ignore")

print("New Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())