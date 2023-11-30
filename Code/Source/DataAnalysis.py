import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3i) The categorical features are hour_of_day, day_of_week, month, holiday, weekday, increase_stock.
# The numerical features are the complement of that.

# 3ii)

data = pd.read_csv('../Data/training_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Get a concise summary of the dataset
print(data.info())

# Get basic statistical details
print(data.describe(include='all'))

# Hourly trend of bicycles. Saved in ('../Plots/hour_against_increase_stock.png')

#sns.countplot(x = 'hour_of_day', hue = 'increase_stock', data = data)
#plt.show()

# Daily trend of bicycles. Saved in ('../Plots/day_of_week_against_increase_stock')

#sns.countplot(x = 'day_of_week', hue = 'increase_stock', data = data)
#plt.show()

# Monthly trend of bicycles. Saved in ('../Plots/month_against_increase_stock')

#sns.countplot(x = 'month', hue = 'increase_stock', data = data)
#plt.show()

# FEATURE ENGINEERING
filtered_data = data[~data['day_of_week'].isin([5, 6])]
sns.countplot(x = 'hour_of_day', hue = 'increase_stock', data = filtered_data)
plt.show()




