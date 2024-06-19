import glob
import math
import os
import sys
import random
import time
import argparse
import re

# Assicurati di avere CARLA nel percorso
try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla

IMG_FOLDER = ""
IMG_WIDTH = 480
IMG_HEIGHT = 480
TOTAL_PHOTOS = 0

def gen_output_dir():
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
        

def get_yaw_from_vector(v):
    return math.degrees(math.atan2(v.y, v.x))

def main():
    parser = argparse.ArgumentParser(description="CARLA Data Collection Script")
    parser.add_argument(
        "--host",
        default="localhost",
        help="IP of the host CARLA Simulator (default: localhost)",
    )
    parser.add_argument(
        "--port",
        default=2000,
        type=int,
        help="TCP port of CARLA Simulator (default: 2000)",
    )
    parser.add_argument(
        "--map",
        default="Town12",
        help="Load a new map, use --list to see available maps",
    )
    parser.add_argument(
        "--weather",
        default="ClearNoon",
        help="Set weather preset, use --list to see available presets",
    )
    parser.add_argument(
        "--res", default="480x480", help="Resolution of the camera (default: 480x480)"
    )
    parser.add_argument(
        "--fps",
        default=0,
        type=float,
        help="Set fixed FPS, zero for variable FPS (default: 30)",
    )
    parser.add_argument(
        "--delta-seconds",
        default=0.05,
        type=float,
        help="Set fixed delta seconds, zero for variable frame rate (default: 0.05)",
    )
    parser.add_argument(
        "--timeout",
        default=60.0,
        type=float,
        help="Timeout for CARLA client (default: 60 seconds)",
    )
    parser.add_argument(
        "--spectator-speed",
        default=100.0,
        type=float,
        help="Speed of the spectator (default: 10.0)",
    )
    parser.add_argument(
        "--camera",
        default=True,
        type= bool,
        help="Enable camera sensor (default: True)",
    )
    parser.add_argument(
        '-r', '--reload-map',
        action='store_true',
        help='reload current map'
    )
    parser.add_argument(
        '-f', '--first',
        action='store_true',
        help='load the entire map'
    )
        
    args = parser.parse_args() 
    
    IMG_WIDTH, IMG_HEIGHT = map(int, args.res.split("x"))

    client = carla.Client(args.host, args.port, worker_threads=1)
    client.set_timeout(args.timeout)

    try:
        if args.first:
            print("First map loading")
            if args.reload_map:
                print("Reloading Map")
                world = client.reload_world()
            elif args.map is not None:
                world = client.load_world(args.map)
                print(f"Mondo creato '{args.map}'")
            
            world = client.get_world()
            time.sleep(5)
            settings = world.get_settings()

            if args.delta_seconds is not None:
                settings.fixed_delta_seconds = args.delta_seconds
            elif args.fps is not None:
                settings.fixed_delta_seconds = (1.0 / args.fps) if args.fps > 0.0 else 0.0

            if args.delta_seconds is not None or args.fps is not None:
                if settings.fixed_delta_seconds > 0.0:
                    print(
                        "set fixed frame rate %.2f milliseconds (%d FPS)"
                        % (
                            1000.0 * settings.fixed_delta_seconds,
                            1.0 / settings.fixed_delta_seconds,
                        )
                    )
                else:
                    print("set variable frame rate.")
                    settings.fixed_delta_seconds = None

            weather_presets = {
                x: getattr(carla.WeatherParameters, x)
                for x in dir(carla.WeatherParameters)
                if re.match("[A-Z].+", x)
            }
            if args.weather in weather_presets:
                world.set_weather(weather_presets[args.weather])
            else:
                print(f"Weather preset '{args.weather}' not found. Using default weather.")

            # settings.fixed_delta_seconds = args.delta_seconds
            # settings.synchronous_mode = True  # Attiva la modalità sincrona
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        else:
            world = client.get_world()
        # Aggiungi il punto di spawn per lo spettatore
        spectator = world.get_spectator()
        # spectator.set_transform(
        #     carla.Transform(carla.Location(x=569.050476, y=3506.865967, z=370.496704)),
        # )

        # Ottieni il veicolo
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn_point = carla.Transform(
            carla.Location(x=569.050476, y=3501.865967, z=370.496704),
            carla.Rotation(yaw=-90), #90
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # print("Vehicle spawned at:", vehicle.get_transform())
        # time.sleep(10)
        spectator_speed = args.spectator_speed
        gen_output_dir()
        print("Attaching camera...")
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f"{IMG_WIDTH}")
        camera_bp.set_attribute('image_size_y', f"{IMG_HEIGHT}")
        camera_transform = carla.Transform(carla.Location(x=1, z=2, y= 1)) #x=1.5, z=2.4 
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        vehicle.set_autopilot(True)
        # print(f"Frame: {camera_sensor.frame}")
        camera_sensor.listen(lambda data: data.save_to_disk(f"{IMG_FOLDER}stop_{data.frame}"))
        time.sleep(60) # TEMPO DI ESECUZIONE IN SECONDI
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Destroying actors...")
        try:
            if camera_sensor is not None:
                print(f"Destroying '{camera_sensor}'")
                camera_sensor.stop()
                camera_sensor.destroy()
        except:
            pass
        try:
            if vehicle is not None:
                print(f"Destroying '{vehicle}'")
                vehicle.destroy()
        except:
            pass
        # try:
        #     if ego_vehicle is not None:
        #         print(f"Destroying '{ego_vehicle}'")
        #         ego_vehicle.destroy()
        # except:
        #     pass


if __name__ == "__main__":
    main()
