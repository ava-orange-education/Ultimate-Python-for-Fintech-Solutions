from google.cloud import bigquery
from google.oauth2 import service_account


account_credentials = service_account.Credentials.from_service_account_file(
    filename="public_data_g.json"

)

big_query_client = bigquery.Client(
    credentials=account_credentials,
    project=account_credentials.project_id,
)



sql_query = (
    'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` '
    'WHERE state = "CA" '
    'LIMIT 10')
big_query_job = big_query_client.query(sql_query)
data_rows = big_query_job.result() 

for data_row in data_rows:
    print(data_row.name)
    
    
data_frame = big_query_job.to_dataframe() 
data_frame.to_csv('public_usa_names_data.csv', index=False,header=True)
print('public data is written in csv')


