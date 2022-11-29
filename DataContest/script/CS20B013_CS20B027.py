import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/sample_submission.csv')
bookings_data = pd.read_csv('../data/bookings_data.csv')
bookings = pd.read_csv('../data/bookings.csv')
hotel_data = pd.read_csv('../data/hotels_data.csv')
customer_data = pd.read_csv('../data/customer_data.csv')
payments_data = pd.read_csv('../data/payments_data.csv')

# convert payment type to numeric using sk preprocessing label encoder
le = LabelEncoder()
le.fit(payments_data['payment_type'])
payments_data['payment_type'] = le.transform(payments_data['payment_type'])

payments_data['payment_type'] = payments_data['payment_type'] + 1

# keep only entries with payment_sequential as 1
payments_data_unique = payments_data[payments_data['payment_sequential'] == 1]
payments_data_repeat = payments_data[payments_data['payment_sequential'] > 1]

# sort payments_data_repeat by payment_sequential
payments_data_repeat = payments_data_repeat.sort_values(by=['payment_sequential'], ascending=True)

# making payment data unique for each booking_id by adding the payments made by other methods to primary payment method
print("Process 1 Running")
columns = ['payment_value', 'payment_installments']

for payment_data_repeat in payments_data_repeat.itertuples():
    booking_id = payment_data_repeat.booking_id
    payment_data_unique = payments_data_unique[payments_data_unique['booking_id'] == booking_id]
    for column in columns:
        new_value = payment_data_unique[column] + payment_data_repeat.__getattribute__(column)
        payments_data_unique.loc[payments_data_unique['booking_id'] == booking_id, column] = new_value
    payments_data_unique.loc[payments_data_unique['booking_id'] == booking_id, 'payment_sequential'] = payment_data_repeat.payment_sequential

bookings_data_new = bookings_data.merge(hotel_data, on='hotel_id', how='left')

# changing the date format to datetime
bookings_data_new['booking_expiry_date'] = pd.to_datetime(bookings_data_new['booking_expiry_date'])

# change to seconds
bookings_data_new['booking_expiry_date'] = bookings_data_new['booking_expiry_date'].astype(np.int64) // 10 ** 9

# split bookings_data into unique and repeat bookings
bookings_data_unique = bookings_data_new[bookings_data_new['booking_sequence_id'] == 1]
bookings_data_repeat = bookings_data_new[bookings_data_new['booking_sequence_id'] > 1]

# sort bookings_data_repeat by booking_sequence_id
bookings_data_repeat = bookings_data_repeat.sort_values(by=['booking_sequence_id'], ascending=True)

# merging bookings_data for each booking_id
print("Process 2 Running")
columns = ['price', 'agent_fees', 'hotel_name_length', 'hotel_description_length', 'hotel_photos_qty', 'booking_expiry_date']

for booking_data_repeat in bookings_data_repeat.itertuples():
    bookings_id = booking_data_repeat.booking_id
    booking_data_unique = bookings_data_unique[bookings_data_unique['booking_id'] == bookings_id]
    for column in columns:
        new_value = booking_data_unique[column] + booking_data_repeat.__getattribute__(column)
        bookings_data_unique.loc[bookings_data_unique['booking_id'] == bookings_id, column] = new_value
    bookings_data_unique.loc[bookings_data_unique['booking_id'] == bookings_id, 'booking_sequence_id'] = booking_data_repeat.booking_sequence_id

# make entries in bookings_data_unique by taking average of the values based in booking_sequence_id
columns = [ 'hotel_name_length', 'hotel_description_length', 'hotel_photos_qty', 'booking_expiry_date']

for column in columns:
    bookings_data_unique[column] = bookings_data_unique[column] / bookings_data_unique['booking_sequence_id']

# merge bookings and bookings_data as bookings_df
bookings_df = pd.merge(bookings, bookings_data_unique, on='booking_id', how='left')

# merge bookings_df and customer_data as bookings_customer_df
bookings_customer_df = pd.merge(bookings_df, customer_data, on='customer_id', how='left')

# merge bookings_hotel_df and payments_data as bookings_payment_df
bookings_payment_df = pd.merge(bookings_customer_df, payments_data_unique, on='booking_id', how='left')

bookings_payment_df.drop(['customer_id'], axis=1, inplace=True)

cat_columns = ['seller_agent_id', 'booking_status', 'country', 'customer_unique_id', 'hotel_id']

for column in cat_columns:
    le = LabelEncoder()
    le.fit(bookings_payment_df[column])
    bookings_payment_df[column] = le.transform(bookings_payment_df[column])
    if column == 'booking_status' or column == 'country':
        bookings_payment_df[column] = bookings_payment_df[column] + 1

# change date columns to seconds
date_columns = ['booking_create_timestamp', 'booking_approved_at', 'booking_checkin_customer_date']

for column in date_columns:
    bookings_payment_df[column] = pd.to_datetime(bookings_payment_df[column])
    # change to seconds
    bookings_payment_df[column] = bookings_payment_df[column].astype(np.int64) // 10 ** 9

# change approved-at to approved_at - create_timestamp
bookings_payment_df['booking_approved_at'] = bookings_payment_df['booking_approved_at'] - bookings_payment_df['booking_create_timestamp']

# change expiry to expiry - checkin
bookings_payment_df['booking_expiry_date'] = bookings_payment_df['booking_expiry_date'] - bookings_payment_df['booking_checkin_customer_date']

# create new column for expiry - create
bookings_payment_df['booking_expiry_create'] = bookings_payment_df['booking_expiry_date'] - bookings_payment_df['booking_approved_at']

# take all columns
columns = bookings_payment_df.columns

# remove booking_id
columns = columns.drop(['booking_id'])

# change all null or nan values to mean of respective columns
for column in columns:
    mean = bookings_payment_df[column].mean()
    bookings_payment_df[column].fillna(mean, inplace=True)

# scale date columns using StandardScaler
date_columns = ['booking_approved_at', 'booking_expiry_date']

scaled_columns = StandardScaler().fit_transform(bookings_payment_df[date_columns])

bookings_payment_df[date_columns] = scaled_columns

# assert no null values
assert bookings_payment_df.isnull().sum().sum() == 0

# train_booking_df contains bookings_df with booking_id in train_data
train_booking_df = bookings_payment_df[bookings_payment_df['booking_id'].isin(train_data['booking_id'])]

# create X_train and Y_train
train_booking_df = train_booking_df.sort_values(by=['booking_id'])
X_train = train_booking_df.drop(['booking_id'], axis=1)
train_data = train_data.sort_values(by=['booking_id'])

# take only unique values
train_data = train_data.drop_duplicates(subset=['booking_id'])
Y_train = train_data['rating_score']

params = {
    'early_stopping': False, 
    'l2_regularization': 0.1, 
    'learning_rate': 0.04, 'loss': 
    'squared_error', 'max_depth': 5, 
    'max_iter': 400, 
    'max_leaf_nodes': 15, 
    'validation_fraction': 0.2
}

# use best parameters to train model
print("Training")
# use best params
model = HistGradientBoostingRegressor(**params)
# fit model
model.fit(X_train, Y_train)

test_booking_df = bookings_payment_df[bookings_payment_df['booking_id'].isin(test_data['booking_id'])]

# create X_test
test_booking_df = test_booking_df.sort_values(by=['booking_id'])
X_test = test_booking_df.drop(['booking_id'], axis=1)

Y_test_pred = model.predict(X_test)

# prepare submission file
submission = pd.DataFrame()
submission['booking_id'] = test_booking_df['booking_id']
submission['rating_score'] = Y_test_pred

# change ratings below 0 to 0 and above 5 to 5
submission['rating_score'] = submission['rating_score'].apply(lambda x: 1 if x < 1 else x)
submission['rating_score'] = submission['rating_score'].apply(lambda x: 5 if x > 5 else x)

print("Making File")
submission.to_csv('CS20B013_CS20B027_main.csv', index=False)
print("Done")