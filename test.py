import asyncio
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        print(d)

asyncio.run(scan())


#{"sensors":[0,10,20,30,40],"timestamp":143622}
#{"sensors":[0,10,20,30,40],"timestamp":143672}
{0,10,20,30,40}