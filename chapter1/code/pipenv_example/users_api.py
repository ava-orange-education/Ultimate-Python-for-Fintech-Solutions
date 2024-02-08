import requests

response = requests.get('https://reqres.in/api/users?page=2')

print ("users list json",response.json())
