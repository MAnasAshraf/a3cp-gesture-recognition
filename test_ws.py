import asyncio
import websockets

async def test():
    async with websockets.connect("ws://localhost:8000/ws/camera") as ws:
        await ws.send(b"hello")
        await asyncio.sleep(1)

asyncio.run(test())
