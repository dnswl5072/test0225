import requests

temperature = 0
pressure = 0
load = 0
production_rate = 0

scoring_uri = "<your web service URI>"  # TODO: Replace with your scoring uri
data = {"data": [[temperature, pressure, load, production_rate]]}  # TODO: Replace with appropriate values
headers = {"Content-Type": "application/json"}

resp = requests.post(scoring_uri, json=data, headers=headers)
resp.raise_for_status()
print(resp.text)
