import httpx
import asyncio

async def download_file(url: str, dest_path: str):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

asyncio.run(download_file("http://localhost:8000/get/mybucket/image.jpg", "newimage.jpg"))
