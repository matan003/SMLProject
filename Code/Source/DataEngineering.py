import pandas as pd

# Feature engineering

df = pd.read_csv('../Data/training_data.csv')

def is_weekday_peak_hour(row):
    hour = row['hour_of_day']
    day = row['day_of_week']
    return 1 if (8 <= hour <= 9 or 15 <= hour <= 19) and (0 <= day <= 4) else 0

df['is_peak_hour'] = df.apply(is_weekday_peak_hour, axis = 1)

column_order = df.columns.to_list()
column_order.insert(-1, column_order.pop(column_order.index('is_peak_hour')))
df = df[column_order] # It should be inserted as the second last feature.

# Change the increase_stock to 0 and 1
df['increase_stock'] = df['increase_stock'].map({'high_bike_demand': 1, 'low_bike_demand': 0})

df.to_csv('../Data/training_data_engineer_with_is_peak_hour.csv', index = False)
