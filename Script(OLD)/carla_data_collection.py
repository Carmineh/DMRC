import glob
import math
import os
import sys
import carla
import random
import time
import argparse
import re

import pygame

# Assicurati di avere CARLA nel percorso
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Creazione della cartella di output se non esiste [./output_data]
directory_name = "output_data"
current_dir = os.getcwd()
if os.path.exists(directory_name):
    os.rmdir(directory_name)
    print(f"Directory '{directory_name}' deleted")
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' created")
else:
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' already exists in '{current_dir}'")

IMG_FOLDER = f"{current_dir}\\{directory_name}\\"
print(IMG_FOLDER)
IMG_WIDTH = 480
IMG_HEIGHT = 480

TOTAL_PHOTOS = 0

def get_yaw_from_vector(v):
    return math.degrees(math.atan2(v.y, v.x))

def direziona_fotocamera_verso_segnale(camera, segnale):
    camera_location = camera.get_transform().location
    segnale_location = segnale.get_transform().location

    direction = segnale_location - camera_location
    yaw = get_yaw_from_vector(direction)

    # Imposta la rotazione della fotocamera
    camera_transform = camera.get_transform()
    camera_transform.rotation.yaw = yaw
    camera.set_transform(camera_transform)

def scatta_foto(camera_sensor):
    # La callback per salvare l'immagine
    def salva_immagine(image):
        image.save_to_disk(IMG_FOLDER + "%08d" % image.frame)

    # Imposta la callback per il sensore di fotocamera
    camera_sensor.listen(salva_immagine)

def main():
    parser = argparse.ArgumentParser(description="CARLA Data Collection Script")
    parser.add_argument('--host', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    parser.add_argument('--port', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    parser.add_argument('--map', default='Town12', help='Load a new map, use --list to see available maps')
    parser.add_argument('--weather', default='ClearNoon', help='Set weather preset, use --list to see available presets')
    parser.add_argument('--res', default='480x480', help='Resolution of the camera (default: 480x480)')
    parser.add_argument('--fps', default=15, type=float, help='Set fixed FPS, zero for variable FPS (default: 30)')
    parser.add_argument('--delta-seconds', default=0.05, type=float, help='Set fixed delta seconds, zero for variable frame rate (default: 0.05)')
    parser.add_argument('--timeout', default=60.0, type=float, help='Timeout for CARLA client (default: 60 seconds)')
    parser.add_argument('--spectator-speed', default=100.0, type=float, help='Speed of the spectator (default: 10.0)')
    parser.add_argument('--disable-cache', action='store_true', help='Disable the usage of cache')
    args = parser.parse_args()

    quality_level = 'Low'

    # Apply the quality level setting
    os.environ['CARLA_QUALITY_LEVEL'] = quality_level

    disable_cache = args.disable_cache
        
    # Disabling cache by setting environment variable
    if disable_cache:
        os.environ['CARLA_DISABLE_CACHE'] = '1'
        print("Cache is disabled.")

    IMG_WIDTH, IMG_HEIGHT = map(int, args.res.split('x'))

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    vehicle = None
    ego_vehicle = None
    camera_sensor = None

    try:
        if args.map is not None:
            world = client.load_world(args.map)
            world = client.get_world()
        else:
            world = client.get_world()
        print("Mondo creato")
        time.sleep(5)
        settings = world.get_settings()

        if args.delta_seconds is not None:
            settings.fixed_delta_seconds = args.delta_seconds
        elif args.fps is not None:
            settings.fixed_delta_seconds = (1.0 / args.fps) if args.fps > 0.0 else 0.0

        if args.delta_seconds is not None or args.fps is not None:
            if settings.fixed_delta_seconds > 0.0:
                print('set fixed frame rate %.2f milliseconds (%d FPS)' % (
                    1000.0 * settings.fixed_delta_seconds,
                    1.0 / settings.fixed_delta_seconds))
            else:
                print('set variable frame rate.')
                settings.fixed_delta_seconds = None

        weather_presets = {x: getattr(carla.WeatherParameters, x) for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)}
        if args.weather in weather_presets:
            world.set_weather(weather_presets[args.weather])
        else:
            print(f"Weather preset '{args.weather}' not found. Using default weather.")

        # settings.fixed_delta_seconds = args.delta_seconds
        # settings.synchronous_mode = True  # Attiva la modalità sincrona
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        # Aggiungi il punto di spawn per lo spettatore
        time.sleep(10)
        print("Sono trascorsi 10 secondi, ora modifica il mio punto di vista")
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=569.050476, y=3506.865967, z=370.496704)))

        # Verifica se la posizione dello spettatore è stata impostata correttamente
        time.sleep(1)
        print("Spectator position after setting:", spectator.get_transform())

        # Ottieni il veicolo
        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_point = carla.Transform(carla.Location(x=569.050476, y=3501.865967, z=370.496704), carla.Rotation(yaw=-90))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned at:", vehicle.get_transform())

        # Attacca la fotocamera al veicolo
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f"{IMG_WIDTH}")
        camera_bp.set_attribute('image_size_y', f"{IMG_HEIGHT}")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Modifica x, y e z secondo le tue esigenze
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        scatta_foto(camera_sensor)

        spectator_speed = args.spectator_speed
        
        def print_vehicle_position(spectator):
            print("Spectator position after setting:", spectator.get_transform())

        # Loop principale per aggiornare la posizione del veicolo
        while True:
            print_vehicle_position(spectator)
            time.sleep(1)  # Aggiorna ogni secondo

        # # Filtra solo i segnali stradali
        # segnali_stradali = world.get_actors().filter('traffic.stop')

        # # Direziona la fotocamera verso i segnali stradali e scatta le foto
        # for segnale in segnali_stradali:
        #     direziona_fotocamera_verso_segnale(camera_sensor, segnale)


        # Attendi qualche secondo per catturare le immagini
        #time.sleep(5)

    except Exception as e:
        print(f"An error occurred: {e}")

    # finally:
    #     # Distruggi gli attori creati
    #     try:
    #         if camera_sensor is not None:
    #             camera_sensor.stop()
    #             camera_sensor.destroy()
    #     except:
    #         pass
    #     try:
    #         if vehicle is not None:
    #             vehicle.destroy()
    #     except:
    #         pass
    #     try:
    #         if ego_vehicle is not None:
    #             ego_vehicle.destroy()
    #     except:
    #         pass

if __name__ == '__main__':
    main()
