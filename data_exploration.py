
# Step 3.3.1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 3.3.2: Load the AI4I 2020 dataset
df = pd.read_csv("data/ai4i2020.csv")

# Step 3.3.3: Display the first five records
print("First five records:")
print(df.head())

# Step 3.3.4: Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Step 3.3.5: Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Step 3.3.6: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3.3.7: Display dataset information
print("\nDataset Information:")
df.info()


# Step 4.1.1: Count machine failures
print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())

# Step 4.1.2: Calculate failure percentages
print("\nMachine Failure Percentages:")
print(df["Machine failure"].value_counts(normalize=True) * 100)

# Step 4.1.3: Count different fault types
fault_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\nFault Type Distribution:")
print(df[fault_columns].sum())

# Step 4.1.4: Visualize machine failures
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# Step 4.2.1: Define the fault types
fault_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

fault_names = [
    "Tool Wear Failure",
    "Heat Dissipation Failure",
    "Power Failure",
    "Overstrain Failure",
    "Random Failure"
]

# Step 4.2.2: Count each fault type
fault_counts = df[fault_columns].sum()

# Step 4.2.3: Display fault counts
print("\nFault Type Counts:")
print(fault_counts)

# Step 4.2.4: Visualize fault type distribution
plt.figure(figsize=(10, 5))

sns.barplot(
    x=fault_names,
    y=fault_counts.values
)

plt.title("Distribution of Industrial Equipment Fault Types")
plt.xlabel("Fault Type")
plt.ylabel("Number of Fault Occurrences")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()
plt.show()
