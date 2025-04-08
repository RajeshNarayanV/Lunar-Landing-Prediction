import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\finalized dataset.csv")

# Step 2: Data Cleaning
# Parse Launch and Arrival dates
df['Launch'] = pd.to_datetime(df['Launch'], format='%d-%m-%Y', errors='coerce')
df['Arrival'] = pd.to_datetime(df['Arrival'], format='%d-%m-%Y', errors='coerce')

# Fill missing Arrival dates with NaT (Not a Time)
df['Arrival'].fillna(pd.NaT, inplace=True)

# Step 3: Feature Creation
# Calculate Mission Duration in days
df['Mission_Duration'] = (df['Arrival'] - df['Launch']).dt.days

# Fill missing Mission_Duration with 0 (for missions that did not arrive)
df['Mission_Duration'].fillna(0, inplace=True)

# Extract Launch_Year and Arrival_Year
df['Launch_Year'] = df['Launch'].dt.year
df['Arrival_Year'] = df['Arrival'].dt.year

# Fill missing Arrival_Year with 0
df['Arrival_Year'].fillna(0, inplace=True)

# Create a binary target variable (Outcome_Binary)
df['Outcome_Binary'] = df['Outcome'].apply(lambda x: 1 if 'Successful' in str(x) else 0)

# Step 4: Feature Selection
# Select features and target
features = ['Nation', 'Type', 'Mission_Duration', 'Launch_Year', 'Arrival_Year']
X = df[features]
y = df['Outcome_Binary']

# Step 5: Save the Engineered Dataset
# Save the engineered dataset to a new CSV file
df.to_csv(r"C:\Users\Admin\Downloads\finalized_dataset_engineered.csv", index=False)

print("Feature engineering completed! Engineered dataset saved.")