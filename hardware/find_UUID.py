# JUNK FILE
import asyncio
from bleak import BleakClient

ESP32_ADDRESS = "YOUR_MAC_HERE"

async def explore():
    async with BleakClient(ESP32_ADDRESS) as client:
        print("Connected:", await client.is_connected())
        
        services = await client.get_services()
        for service in services:
            print("\nService:", service.uuid)
            for char in service.characteristics:
                print("  Characteristic:", char.uuid)
                print("    Properties:", char.properties)

asyncio.run(explore())
