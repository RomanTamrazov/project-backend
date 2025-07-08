import requests

# with open('testfile.txt', 'rb') as f:
#     data = f.read()

files={
   'file': ('image.jpg', open('image.jpg', 'rb'))
}

response = requests.post('http://127.0.0.1:8000/upload', files=files)
print(response.content)