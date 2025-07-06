import requests

# Your test data
data = {
    "sensor_id": "sensor_1",
    "vehicle_count": 25,
    "wait_time": 45,
    "congestion_level": "medium"
}

# The URL of your backend API
url = "http://127.0.0.1:5000/api/sensor-data"

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error sending data:", e)