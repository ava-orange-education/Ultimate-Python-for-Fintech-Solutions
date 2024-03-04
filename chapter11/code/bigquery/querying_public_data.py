from google.cloud import bigquery
from google.oauth2 import service_account


account_credentials = service_account.Credentials.from_service_account_file(
    filename="public_data_g.json"

)

big_query_client = bigquery.Client(
    credentials=account_credentials,
    project=account_credentials.project_id,
)

public_data = big_query_client.dataset('cms_medicare', project='bigquery-public-data')

print([data_value.table_id for data_value in big_query_client.list_tables(public_data)])