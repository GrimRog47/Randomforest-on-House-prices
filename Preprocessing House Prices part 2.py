import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
house_prices = pd.read_csv(r'U:\New folder\Preprocess\DataSets\house price\cleaned_house_prices_v3.csv')
print('\nDataset loaded successfully')

# log transform the amount variable
house_prices['log_amount'] = np.log(house_prices['Amount(in rupees)'])
print('\nLog transformation applied to Amount(in rupees)')

# Plot histogram of log-transformed amount
plt.figure(figsize=(10, 6))
plt.hist(house_prices['log_amount'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Log-Transformed House Prices')
plt.xlabel('Log(Amount in rupees)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
print('\nHistogram of log-transformed house prices displayed')

# log transform the carpet area variable
house_prices['log_carpet_area'] = np.log(house_prices['Carpet Area'])
print('\nLog transformation applied to Carpet Area(in sqft)')
# Plot histogram of log-transformed carpet area
plt.figure(figsize=(10, 6))
plt.hist(house_prices['log_carpet_area'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Histogram of Log-Transformed Carpet Area')
plt.xlabel('Log(Carpet Area in sqft)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
print('\nHistogram of log-transformed carpet area displayed')

# drop columns that are no longer needed
house_prices.drop(columns=['Super Area', 'Amount(in rupees)','Carpet Area','Status'], inplace=True)
print('\nDropped Super Area, Amount(in rupees), and Carpet Area columns from the dataset')

# hot encode transaction and ownership columns
house_prices = pd.get_dummies(house_prices, columns=['Transaction', 'Ownership'], drop_first=True)
print('\nOne-hot encoding applied to Transaction and Ownership columns')

# one hot encode facing column
house_prices = pd.get_dummies(house_prices, columns=['facing'], drop_first=True)
print('\nOne-hot encoding applied to Facing column')

# target encode location and society columns
location_mean_price = house_prices.groupby('location')['log_amount'].mean()
house_prices['Location_TE'] = house_prices['location'].map(location_mean_price)
print('\nTarget encoding applied to Location column')
society_mean_price = house_prices.groupby('Society')['log_amount'].mean()
house_prices['Society_TE'] = house_prices['Society'].map(society_mean_price)
print('\nTarget encoding applied to Society column')
# drop original location and society columns
house_prices.drop(columns=['location', 'Society'], inplace=True)
print('\nDropped original Location and Society columns from the dataset')

# target encode overlooking column
overlooking_mean_price = house_prices.groupby('overlooking')['log_amount'].mean()
house_prices['Overlooking_TE'] = house_prices['overlooking'].map(overlooking_mean_price)
print('\nTarget encoding applied to overlooking column')
# drop original overlooking column
house_prices.drop(columns=['overlooking'], inplace=True)
print('\nDropped original Overlooking column from the dataset')

# ordinal encode furnishing column
furnishing_mapping = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}
house_prices['Furnishing_Ord'] = house_prices['Furnishing'].map(furnishing_mapping)
print('\nOrdinal encoding applied to Furnishing column')

# seprate digits and text from car parking column
def extract_parking_info(parking_str):
    if pd.isna(parking_str):
        return 0, 'No Parking'
    parts = parking_str.split()
    num_parking = int(parts[0]) if parts[0].isdigit() else 0
    parking_type = ' '.join(parts[1:]) if len(parts) > 1 else 'No Parking'
    return num_parking, parking_type
house_prices[['Car Parking Count', 'Parking Type']] = house_prices['Car Parking'].apply(
    lambda x: pd.Series(extract_parking_info(x)))
house_prices.drop(columns=['Car Parking'], inplace=True)
print('\nExtracted number and type of car parking from Car Parking column')

# ordinal encode parking type column
# remove commas from parking type
house_prices['Parking Type'] = house_prices['Parking Type'].str.replace(',', '')
parking_type_mapping = {'No Parking': 0, 'Open': 1, 'Covered': 2}
house_prices['Parking_Type_Ord'] = house_prices['Parking Type'].map(parking_type_mapping)
print('\nOrdinal encoding applied to Parking Type column')
# drop original parking type column
house_prices.drop(columns=['Parking Type'], inplace=True)
print('\nDropped original Parking Type column from the dataset')

# drop original furnishing column
house_prices.drop(columns=['Furnishing'], inplace=True)
print('\nDropped original Furnishing column from the dataset')

# cap bhk at 10
house_prices['BHK'] = house_prices['BHK'].apply(lambda x: min(x, 10))
print('\nCapped BHK values at 10')
# cap car parking count at 5
house_prices['Car Parking Count'] = house_prices['Car Parking Count'].apply(lambda x: min(x, 5))
print('\nCapped Car Parking Count values at 5')

# reorder log amount to last and log carpet area to last
cols = house_prices.columns.tolist()
cols.append(cols.pop(cols.index('log_carpet_area')))
cols.append(cols.pop(cols.index('log_amount')))
house_prices = house_prices[cols]
print('\nReordered columns to place log_amount and log_carpet_area at the end')
# check final dataset info
print('\nFinal dataset info:')
print(house_prices.info())
# check for missing values
print('\nMissing values in each column:')
print(house_prices.isnull().sum())

# save the preprocessed dataset
house_prices.to_csv(r'U:\New folder\Preprocess\DataSets\house price\Cleaned_house_prices_v4.csv', index=False)
print('\nPreprocessed dataset saved successfully')