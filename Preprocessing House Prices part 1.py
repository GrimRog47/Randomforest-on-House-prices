import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
house_prices = pd.read_csv(r'U:\New folder\Preprocess\DataSets\house price\RAW_house_prices.csv')
print('\nDataset loaded successfully')

# drop unnecessary columns
house_prices.drop(columns=['Index', 'Price (in rupees)', 'Description', 'Dimensions', 'Plot Area'], inplace=True)
print('\nUnnecessary columns dropped successfully')

# seperate digit and text from 'Carpet Area' column
house_prices['Carpet Area (in sqft)'] = house_prices['Carpet Area'].str.extract(r'(\d+)', expand=False).astype(float)
house_prices['Carpet Area Unit'] = house_prices['Carpet Area'].str.extract(r'([a-zA-Z\s]+)', expand=False).str.strip()
print(house_prices[['Carpet Area', 'Carpet Area (in sqft)', 'Carpet Area Unit']].head())
# find distinct values in 'Carpet Area Unit' column
print(house_prices['Carpet Area Unit'].unique())
# Conversion factors to Square Feet (Adjust for your specific region)
conversion_map = {
    'sqft': 1.0,
    'sqm': 10.764,        # 1 Square Meter ≈ 10.764 sqft
    'sqyrd': 9.0,
    'yards': 9.0,         # 1 Square Yard = 9 sqft
    'marla': 272.25,      # Standard Marla = 272.25 sqft
    'kanal': 5445.0,      # 1 Kanal = 20 Marlas = 5,445 sqft
    'acre': 43560.0,      # 1 Acre = 43,560 sqft
    'bigha': 9072.0,      # 1 Bigha ≈ 9,072 sqft (varies by region)
    'guntha': 1089.0,     # 1 Guntha ≈ 1,089 sqft
    'ground': 2400,        # assuming 'ground' means sqft
    'cent': 435.6         # assuming 'cent' means 435.6 sqft
}
# Create the conversion multiplier column
house_prices['multiplier'] = house_prices['Carpet Area Unit'].map(conversion_map).fillna(1.0)
# Calculate final standardized area
house_prices['Carpet Area (in sqft)'] = house_prices['Carpet Area (in sqft)'] * house_prices['multiplier']
# Replace the original 'Carpet Area' with the standardized one
house_prices['Carpet Area in sqft'] = house_prices['Carpet Area (in sqft)']
# Drop the helper columns when finished
house_prices = house_prices.drop(columns=['Carpet Area (in sqft)', 'Carpet Area Unit', 'multiplier'])
house_prices['Carpet Area'] = house_prices['Carpet Area in sqft']
house_prices.drop(columns=['Carpet Area in sqft'], inplace=True)
print('\nCarpet Area Converted to sqft successfully')

# impute missing values with mode for categorical columns having less than 10% missing values
categorical_cols = house_prices.select_dtypes(include=['object']).columns
for col in categorical_cols:
    missing_percentage = house_prices[col].isnull().mean()
    if missing_percentage < 0.1:
        mode_value = house_prices[col].mode()[0]
        house_prices[col].fillna(mode_value, inplace=True)
# impute missing values with median for numerical columns having less than 10% missing values
numerical_cols = house_prices.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    missing_percentage = house_prices[col].isnull().mean()
    if missing_percentage < 0.1:
        median_value = house_prices[col].median()
        house_prices[col].fillna(median_value, inplace=True)
print('\n simple imputation of missing values completed successfully for columns having less than 10% missing values')

# separate digit and text from 'Amount(in rupees)' column
house_prices['Amount value'] = house_prices['Amount(in rupees)'].str.extract(r'(\d+)', expand=False).astype(float)
house_prices['Amount Unit'] = house_prices['Amount(in rupees)'].str.extract(r'([a-zA-Z\s]+)', expand=False).str.strip()
print(house_prices[['Amount(in rupees)', 'Amount value', 'Amount Unit']].head())
# Conversion factors to integral rupees
conversion_map = {
    'Call for Price': np.nan,
    'Lac': 100000,        # 1 Lac ≈ 100,000
    'Cr': 10000000,        # 1 Cr = 10,000,000
}
# Create the conversion multiplier column
house_prices['multiplier'] = house_prices['Amount Unit'].map(conversion_map).fillna(1.0)
# Calculate final standardized amount
house_prices['Amount value'] = house_prices['Amount value'] * house_prices['multiplier']
# Drop the helper columns when finished
house_prices['Amount(in rupees)'] = house_prices['Amount value']
house_prices = house_prices.drop(columns=['Amount value', 'Amount Unit', 'multiplier'])
# drop missing values in 'Amount(in rupees)' column
house_prices.dropna(subset=['Amount(in rupees)'], inplace=True)
print('\nAmount(in rupees) Converted to integral rupees successfully')

# separate digit and text from 'Super Area' column
house_prices['Super Area (in sqft)'] = house_prices['Super Area'].str.extract(r'(\d+)', expand=False).astype(float)
house_prices['Super Area Unit'] = house_prices['Super Area'].str.extract(r'([a-zA-Z\s]+)', expand=False).str.strip()
# FIND distinct values in 'Super Area Unit' column
print(house_prices['Super Area Unit'].unique())
# conversion map for 'Super Area Unit' column
conversion_map = {
    'sqft': 1.0,
    'sqm': 10.764,        # 1 Square Meter ≈ 10.764 sqft
    'sqyrd': 9.0,
    'marla': 272.25,      # Standard Marla = 272.25 sqft
    'kanal': 5445.0,      # 1 Kanal = 20 Marlas = 5,445 sqft
    'acre': 43560.0,      # 1 Acre = 43,560 sqft
    'ground': 2400,        # assuming 'ground' means sqft
    'aankadam': 72.25         # assuming 'aankadam' means 72.25 sqft
}
# Create the conversion multiplier column
house_prices['multiplier'] = house_prices['Super Area Unit'].map(conversion_map).fillna(1.0)
# Calculate final standardized area
house_prices['Super Area (in sqft)'] = house_prices['Super Area (in sqft)'] * house_prices['multiplier']
# Replace the original 'Super Area' with the standardized one
house_prices['Super Area'] = house_prices['Super Area (in sqft)']
# Drop the helper columns when finished
house_prices = house_prices.drop(columns=['Super Area (in sqft)', 'Super Area Unit', 'multiplier'])
print('\nSuper Area Converted to sqft successfully')

# carpet area missing values indicator
house_prices['Carpet Area Missing'] = house_prices['Carpet Area'].isnull().astype(int)
# if Carpet Area is missing, use super area to fill it
if house_prices['Carpet Area'].isnull().any():
    house_prices['Carpet Area'].fillna(house_prices['Super Area']*0.75, inplace=True)
# if super area is missing, use carpet area to fill it
if house_prices['Super Area'].isnull().any():
    house_prices['Super Area'].fillna(house_prices['Carpet Area']/0.75, inplace=True)
# use the median grouped by location to fill any remaining missing values
house_prices['Carpet Area'].fillna(house_prices.groupby('location')['Carpet Area'].transform('median'), inplace=True)
if house_prices['Super Area'].isnull().any() and house_prices['Carpet Area'].isnull().sum()<=30:
    house_prices['Super Area'].fillna(house_prices.groupby('location')['Super Area'].transform('median'), inplace=True)
print('\nImputation of Carpet Area and Super Area completed successfully')

# seprate digits from 'Title' column
house_prices['BHK'] = house_prices['Title'].str.extract(r'(\d+)', expand=False).astype('Int64')
# remove digits and 'BHK Ready to Occupy Flat for sale' and if 'in' exist remove that too from 'Title' column
house_prices['Title'] = house_prices['Title'].str.replace(r'(\d+\s*BHK\s*Ready to Occupy Flat for sale)', '', regex=True)
house_prices['Title'] = house_prices['Title'].str.replace(r'\bin\b', '', regex=True).str.strip()
# remove strings in title that are also present in location column
location_set = set(house_prices['location'].unique())
def clean_title(title):
    for loc in location_set:
        title = title.replace(loc, '').strip()
    return title
house_prices['Title'] = house_prices['Title'].apply(clean_title)
# fill missing values in 'Society' column with Title column values
house_prices['Society'].fillna(house_prices['Title'], inplace=True)
# drop title and column
house_prices.drop(columns=['Title'], inplace=True)
print('\nBHK column created and Title column cleaned successfully')

# FILL MISSING VALUES in bhk with mode grouped by bathroom
pd.to_numeric(house_prices['BHK'], errors='coerce').astype('Int64')
house_prices['BHK'].fillna(house_prices.groupby('Bathroom')['BHK'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan), inplace=True)
print('\nImputation of BHK completed successfully')

from scipy.stats import chi2_contingency
# check missingness of ownership dependent on other categorical columns
categorical_cols = house_prices.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Ownership':
        contingency_table = pd.crosstab(house_prices[col], house_prices['Ownership'].isnull())
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        print(f'Chi-square test between Ownership and {col}: p-value = {p}')
def fill_mode_nan(x):
    mode = x.mode()
    if not mode.empty:
        return x.fillna(mode.iloc[0])
    else:
        return x.fillna(np.nan)
# fill missing values in 'Ownership' column with mode grouped by society
house_prices['Ownership'].fillna(house_prices.groupby('Society')['Ownership'].transform(fill_mode_nan), inplace=True)
#fill remaining missing values in 'Ownership' column with mode grouped by location
house_prices['Ownership'].fillna(house_prices.groupby('location')['Ownership'].transform(fill_mode_nan), inplace=True)
print('\nImputation of Ownership completed successfully')

# convert 'Balcony' column to numeric
house_prices['Balcony'] = pd.to_numeric(house_prices['Balcony'], errors='coerce').astype('Int64')
# impute missing values in balcony column with median grouped by bhk
house_prices['Balcony'].fillna(house_prices.groupby('BHK')['Balcony'].transform('median').astype('Int64'), inplace=True)
house_prices['Balcony'].fillna(house_prices.groupby('Society')['Balcony'].transform('median').astype('Int64'), inplace=True)
if house_prices['Balcony'].isnull().any():
    house_prices['Balcony'].fillna(house_prices['Balcony'].median(), inplace=True)
house_prices.drop(columns=['Bathroom'], inplace=True)
print('\nDropping Bathroom Column\nImputation of Balcony completed successfully')

# fill facing with mode grouped by society
house_prices['facing'] = house_prices.groupby('Society')['facing'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'North'))
# fill remaining with mode grouped by location
house_prices['facing'] = house_prices.groupby('location')['facing'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'North'))
print('\nimputation of Facing completed successfully')
# fill overlooking with mode grouped by facing and location
house_prices['overlooking'] = house_prices.groupby('facing')['overlooking'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'No'))
house_prices['overlooking'] = house_prices.groupby('location')['overlooking'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'No'))
print('\nimputation of Overlooking completed successfully')
# fill car parking with mode grouped by society and remaining with 0
house_prices['Car Parking'] = house_prices.groupby('Society')['Car Parking'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
house_prices['Car Parking'] = house_prices['Car Parking'].fillna(0)
print('\nimputation of Car Parking completed successfully')
# drop duplicate rows
house_prices = house_prices.drop_duplicates()
print('\nDuplicate rows dropped successfully')

# seperate first 2 digits from floor column
house_prices['Floor number'] = house_prices['Floor'].str.extract(r'(\d+)').astype(float)
house_prices['Floor number'] = house_prices['Floor number'].fillna(house_prices['Floor']).replace({'Ground':0,'Lower Basement':0}).astype(float)
house_prices['Floor number'] = pd.to_numeric(house_prices['Floor number'], errors='coerce').astype(int)
print('\n Floor number column created successfully')
# seperate last 3 digits from floor column
house_prices['Total floors'] = house_prices['Floor'].str.extract(r'(\d+)$').astype(float)
house_prices['Total floors'] = house_prices['Total floors'].fillna(house_prices['Floor']).replace({'Ground':0,'Lower Basement':0}).astype(float)
house_prices['Total floors'] = pd.to_numeric(house_prices['Total floors'], errors='coerce').astype(int)
house_prices.drop('Floor', axis=1, inplace=True)
print('\ntotal floors columns created successfully')

# save the cleaned dataset
house_prices.to_csv(r'U:\New folder\Preprocess\DataSets\house price\cleaned_house_prices_v3.csv', index=False)
print('Cleaned dataset saved successfully.')
