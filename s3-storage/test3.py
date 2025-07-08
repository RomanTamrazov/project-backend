import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder

url = "http://localhost:8000/task/upload/image"

# 1. Prepare the task JSON (must match TaskRequestModel)
task_data = {
    "category": "attitude",
    "data_json": {"test": "hi"},
    "assigned_user_id": 123
}

multipart_data = MultipartEncoder(
    fields={
        'task': ('', json.dumps(task_data), 'application/json'),  # Empty filename
        'file': ('image.jpg', open('image.jpg', 'rb'), 'image/jpeg'),
        'second_file': ('image2.webp', open('image2.webp', 'rb'), 'image/webp')
    }
)

# 2. Prepare files (required file + optional second file)
files = [
    ('task', (None, json.dumps(task_data), 'application/json')),  # JSON as text part
    ('file', ('image.jpg', open('image.jpg', 'rb'), 'image/jpeg')),  # Required file
    ('second_file', ('image2.webp', open('image2.webp', 'rb'), 'image/webp')),
    # ('second_file', ('image2.jpg', open('image2.jpg', 'rb'), 'image/png'))  # Optional
]

# 3. Send request
response = requests.post(
    url,
    data=multipart_data,
    headers={
        'Content-Type': multipart_data.content_type,
        'Accept': 'application/json'
    }
)

print(response.status_code)
print(response.json())
