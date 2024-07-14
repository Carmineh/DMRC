import glob
import math
import os
import sys
import time
import argparse
import re
import shutil


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
TOTAL_PHOTOS = 190


directory_name = "output_data\\STOP_SIGN"
current_dir = os.getcwd()
if os.path.exists(directory_name):
    # shutil.rmtree(directory_name)
    # print(f"Directory '{directory_name}' deleted")
    # os.makedirs(directory_name)
    # print(f"Directory '{directory_name}' created")
    print("Folder already exists")
else:
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' already exists in '{current_dir}'")

IMG_FOLDER = f"{current_dir}\\{directory_name}\\"

def get_yaw_from_vector(v):
    return math.degrees(math.atan2(v.y, v.x))

def salva_immagine(image):
        global TOTAL_PHOTOS
        TOTAL_PHOTOS += 1
        image.save_to_disk(IMG_FOLDER + "STOP_%d" % TOTAL_PHOTOS)

def scatta_foto(sensor):
    global TOTAL_PHOTOS
    sensor.listen(salva_immagine)
    

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
        action="store_true",
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
        '-p', '--params',
        action='store_true',
        help='load the entire map'
    )
    parser.add_argument(
        '--spectator',
        action='store_true',
        help='move the spectator to the default position '
    )
    parser.add_argument(
        '--sign',
        default="",
        help="Enable camera sensor (default: True)",
    )
    parser.add_argument(
        '--quality',
        action='store_true',
        help='Change engine quality'
    )
    parser.add_argument(
        '--spectator_cords',
        action='store_true',
        help='Start printing spectator cords'
    )
    parser.add_argument(
        '--vehicle',
        action='store_true',
        help='Spawn vehicle with camera and start taking photo'
    )
    
        
    args = parser.parse_args() 
    
    IMG_WIDTH, IMG_HEIGHT = map(int, args.res.split("x"))

    client = carla.Client(args.host, args.port, worker_threads=1)
    client.set_timeout(args.timeout)

    try:
        if args.params:
            print("Loading with parameters...")
            if args.reload_map:
                print("Reloading Map...")
                world = client.reload_world()
            elif args.map:
                map_input = input("Insert the map to load: ")
                print(f"Loading Map '{map_input}'...")
                world = client.load_world(map_input)
                
            world = client.get_world()
            time.sleep(5)
            settings = world.get_settings()
            if args.quality:
                settings.quality_level = carla.QualityLevel.High

            weather_presets = {
                x: getattr(carla.WeatherParameters, x)
                for x in dir(carla.WeatherParameters)
                if re.match("[A-Z].+", x)
            }
            
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            spectator = world.get_spectator()
            if args.spectator:
                print("Moving spectator to default position...")
                spectator = world.get_spectator()
                spectator.set_transform(
                    carla.Transform(carla.Location(x=569.050476, y=3506.865967, z=370.496704)),
                )
            if args.spectator_cords:
                while(True):
                    print(spectator.get_transform())
            
        else:
            world = client.get_world()

        # (x=153, y=306.5, z=0.25) => TESTING 30 => ROTATION: 0°
        # (x=160.303284, y=302.700, z=0.25) => TESTING 60 => -180°
        # (x=14.143657, y=109.610489, z=0.25)  => TESTING 90 => 0°
        # (x=31.344999, y=3798.967041, z=373.927063) => TESTING STOP =>
 
        if args.vehicle:
            vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
            # #x=569.050476, y=3501.865967, z=370.496704 STOP
            spawn_point_12 = carla.Transform(
                carla.Location(x=33.5, y=3797, z=370), 
                carla.Rotation(yaw=90), #90
            )
            #X=160 => 90 | X=60 => 60 | X=35 => 30 | x=569.050476, y=3501.865967, z=370.496704
            spawn_point_02 = carla.Transform(
                carla.Location(x=15, y=109.610489, z=0.25), 
                carla.Rotation(yaw=0), 
            )
            vehicle = world.spawn_actor(vehicle_bp, spawn_point_12)
            vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))
            # print("Vehicle spawned at:", vehicle.get_transform())
            time.sleep(2)
            
            print("Attaching camera...")
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', f"{IMG_WIDTH}")
            camera_bp.set_attribute('image_size_y', f"{IMG_HEIGHT}")
            camera_bp.set_attribute('fov', '30')

            
            camera_transform = carla.Transform(carla.Location(x=1, z=2, y= 1.6), carla.Rotation(yaw=15)) #x=yaw=20 for Town12
            camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            scatta_foto(camera_sensor)
        time.sleep(5)

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
