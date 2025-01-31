import asyncio
import websockets

from PIL import Image, ImageDraw, ImageFilter
from random import randint
import time

#Should be map image
map_image = Image.open('map_image.png')

#Should be any drone placeholder image
drone_image = Image.open('drone.png')

#Placeholder dot image
square_image = Image.open('redsquare.png')

resized_square = square_image.resize((10,10))
resized_drone_img = drone_image.resize((50, 50)) 
modifiable_map_image = map_image

#Input coords based on which map you are using
#Current coords are for competition map
left_longitude = -76.5566927
top_latitude = 38.319353
right_longitude = -76.5399942
bottom_latitude = 38.3128429

map_width, map_height = map_image.size
#Calculate difference for map position calculation
longitude_difference = left_longitude - right_longitude
latitude_difference = top_latitude - bottom_latitude

#Calculate multipliers for map image placement
longitude_multiplier = map_width / longitude_difference
latitude_multiplier = map_height / latitude_difference

#Test coordinates (should be switched to actual inputs)
longitude = -76.548
latitude = 38.317
#Calculate where to place dot
longitude_position = left_longitude - longitude
latitude_position = top_latitude - latitude

#Calculate pixel position on map image
longitude_map_position = (longitude_position * longitude_multiplier) - 25
latitude_map_position = (latitude_position * latitude_multiplier) - 25
longitude_map_position = int(longitude_map_position)
latitude_map_position = int(latitude_map_position)
#Confirm that expected coordinates were found
print(longitude_map_position, latitude_map_position)

#ADD: paste drone image on map at determined coordinates
modifiable_map_image.paste(resized_drone_img, (longitude_map_position, latitude_map_position))
modifiable_map_image.save('modded_map.png')
final_map_image = Image.open('map_image.png')
final_map_image.save('in_progress_map.png')


def map_update():
  #Save updated map image with drone position 
  global modifiable_map_image, final_map_image, longitude_map_position, latitude_map_position
  modifiable_map_image.save('modded_map.png')
  modifiable_map_image = Image.open('in_progress_map.png')
  final_map_image.paste(resized_square, (longitude_map_position+20, latitude_map_position+20))
  final_map_image.save('in_progress_map.png')
  longitude_map_position = (randint(1, 1000))
  latitude_map_position = (randint(1, 500))
  print(longitude_map_position, latitude_map_position)
  modifiable_map_image.paste(resized_drone_img, (longitude_map_position, latitude_map_position))


async def echo(websocket, path=None):
    print("Client connected")
    try:
        async for message in websocket:
            print(f"Received message from client: {message}")
            map_update()
            await websocket.send(f"Server says: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())


