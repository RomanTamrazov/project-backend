import requests

# with open('testfile.txt', 'rb') as f:
#     data = f.read()

files={
   'file': ('image.jpg', open('image.jpg', 'rb'))
}

# response = requests.post('http://127.0.0.1:8000/upload', files=files)
# print(response.content)

DB_USER = "testuser"
DB_PASSWORD = "passwd123"
DB_HOST = "0.0.0.0"
DB_PORT = 5432
DB_NAME = "mydb"

DATABASE_URL = (f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@"
                f"{DB_HOST}:{DB_PORT}/{DB_NAME}")
print(DATABASE_URL)