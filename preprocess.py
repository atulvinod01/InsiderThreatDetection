#preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load and preprocess data
def load_and_preprocess_data():
    # Load logon, device, LDAP, and HTTP data
    logon_df = pd.read_csv('r1/logon.csv')
    device_df = pd.read_csv('r1/device.csv')
    ldap_dfs = [pd.read_csv(f'r1/LDAP/2010-{i:02d}.csv') for i in range(1, 13)]
    http_df = pd.read_csv('r1/http.csv')

    # Merge logon and device data
    merged_df = pd.merge(logon_df, device_df, how='outer', on=['id', 'date', 'user', 'pc'])

    # Merge with LDAP data to get user roles
    for i, ldap_df in enumerate(ldap_dfs, 1):
        suffix = f"_{i}"
        user_id_column = ldap_df['employee_name'].str.split().apply(lambda x: ''.join(part[0].upper() for part in x)) + suffix.zfill(3)

        merged_df = pd.merge(merged_df, ldap_df, how='left', left_on='user', right_on=user_id_column, suffixes=('', suffix))
        merged_df['role'] = merged_df['Role']
        merged_df['role'] = merged_df['Role'].combine_first(merged_df['role'])

    # Drop unnecessary columns
    merged_df.drop(columns=[f'user_id_{i}', f'Email_{i}', f'employee_name_{i}', f'Role_{i}', f'Domain_{i}'], inplace=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df['role'].fillna('NormalUser', inplace=True)

    # Create a copy of the merged data for further processing
    final_df = merged_df.copy()
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df['day_of_week'] = final_df['date'].dt.dayofweek
    final_df['after_hours'] = ((final_df['date'].dt.hour < 9) | (final_df['date'].dt.hour > 17)).astype(int)

    # More HTTP data processing
    http_df.rename(columns={'http://cnet.com': 'url'}, inplace=True)
    merged_df = pd.merge(merged_df, ldap_df, how='left', left_on='user', right_on=user_id_column, suffixes=('', suffix))
    merged_df['role'] = merged_df['Role']
    merged_df['role'] = merged_df['Role'].combine_first(merged_df['role'])
    merged_df.drop(columns=[f'user_id_{i}', f'Email_{i}', f'employee_name_{i}', f'Role_{i}', f'Domain_{i}'], inplace=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df['role'].fillna('NormalUser', inplace=True)

    http_df.rename(columns={'http://cnet.com': 'url'}, inplace=True)
    merged_df = pd.merge(merged_df, ldap_df, how='left', left_on='user', right_on=user_id_column, suffixes=('', suffix))
    merged_df['role'] = merged_df['Role']
    merged_df['role'] = merged_df['Role'].combine_first(merged_df['role'])
    merged_df.drop(columns=[f'user_id_{i}', f'Email_{i}', f'employee_name_{i}', f'Role_{i}', f'Domain_{i}'], inplace=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df['role'].fillna('NormalUser', inplace=True)

    # Create the final DataFrame
    final_df = merged_df.copy()
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df['day_of_week'] = final_df['date'].dt.dayofweek
    final_df['after_hours'] = ((final_df['date'].dt.hour < 9) | (final_df['date'].dt.hour > 17)).astype(int)

    # Final HTTP data processing
    http_df.rename(columns={'http://cnet.com': 'url'}, inplace=True)
    http_df.rename(columns={'01/04/2010 07:08:47': 'date'}, inplace=True)
    http_df.rename(columns={'{M8H9-W9NL75TH-1322KOLO}': 'user'}, inplace=True)
    http_df['date'] = pd.to_datetime(http_df['date'])
    http_df['domain'] = http_df['url'].str.extract(r'^(?:https?://)?(?:www\.)?([^/]+)')

    # Calculate URL counts and merge with the final DataFrame
    url_counts = http_df.groupby(['user', 'date'])['url'].count().reset_index()
    url_counts.rename(columns={'url': 'url_count'}, inplace=True)
    final_df = pd.merge(final_df, url_counts, how='left', left_on=['user', 'date'], right_on=['user', 'date'])
    final_df['url_count'].fillna(0, inplace=True)
    final_df.drop(columns=['user'], inplace=True)

    # Label threats based on certain conditions
    final_df['threat'] = 0
    final_df.loc[final_df['activity_y'] == 'connect', 'threat'] = 1
    abnormal_logon_pattern = ((final_df['date'].dt.hour < 9) | (final_df['date'].dt.hour > 17))
    final_df.loc[abnormal_logon_pattern, 'threat'] = 1
    admin_users = final_df[final_df['Role'] == 'Systems Administrator']['user_id'].unique()
    final_df.loc[final_df['user_id'].isin(admin_users), 'threat'] = 1

    # Select features for clustering
    features = ['day_of_week', 'after_hours', 'url_count']
    X = final_df[features]
    y = final_df['threat']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    global stored_X_test, stored_y_test
    stored_X_test, stored_y_test = X_test, y_test

    return X, X_train, X_test, y_train, y_test, features, final_df

# Load and preprocess data initially
X, X_train, X_test, y_train, y_test, features, final_df = load_and_preprocess_data()
print("Data Preprocessing Completed.")

def process_new_data(new_data):
    # Replace the following lines with the actual preprocessing steps for your new data
    new_data.rename(columns={'new_column_name': 'url'}, inplace=True)
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['domain'] = new_data['url'].str.extract(r'^(?:https?://)?(?:www\.)?([^/]+)')

    # Assuming the same features are used for the model
    new_features = new_data[['day_of_week', 'after_hours', 'device_count', 'url_count', 'email_count', 'attachment_count', 'file_count']]

    return new_features

def generate_alerts():
    pass
