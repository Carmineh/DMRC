import glob
import os
import sys
# import keyboard

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
import random
import time

# Creazione della cartella di output se non esiste [./output_data]
directory_name = "output_data"
current_dir = os.getcwd()
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' created in '{current_dir}'")
else:
    print(f"Directory '{directory_name}' already exists in '{current_dir}'")

IMG_FOLDER = os.path.join(current_dir, directory_name, "/")
IMG_WIDTH = 480
IMG_HEIGHT = 480

TOTAL_PHOTOS = 0


def save_image(image):
    image.save_to_disk(IMG_FOLDER + "%08d" % image.frame)
    TOTAL_PHOTOS += 1
    print(f"Total images saved: {TOTAL_PHOTOS}")


actor_list = []
try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    
    vehicle_blueprint = blueprint_library.find('vehicle.tesla.model3')  # Scelta del modello dell'auto
    ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    vehicle_blueprint.set_attribute('role_name', 'hero')
    ego_bp.set_attribute('role_name', 'hero')
    print (vehicle_blueprint)
    print(ego_bp)
    vehicle_spawn_point = carla.Transform( carla.Location(x = 210.86701172 ,y = 199.44183594, z = 1)) 
    vehicle_spawn_point2 = carla.Transform( carla.Location(x = 212.86701172 ,y = 194.44183594, z = 1)) 
    #(X=21086.701172,Y=19944.183594,Z=30.000000)
    # X = 195 => 19500 | Y = 2 => 200 | Z = 0.01 => 1  coordinataUnreal = coordinataNostra * 100
    # vehicle_spawn_point = carla.Transform( carla.Location(x = 21.264230469 ,y = 0.3, z = 19.477292969))
    
    #bp = blueprint_library.filter("model3")[0]  # Scelta del modello dell'auto
    #vehicle_spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_blueprint, vehicle_spawn_point)
    ego_vehicle = world.spawn_actor(ego_bp, vehicle_spawn_point2)
    print(vehicle)
    print(ego_vehicle)


    items = world.get_actors()
    vehicle_actor = items[len(items) - 1]

    actor_list.append(vehicle)
    actor_list.append(ego_vehicle)

    sensor_blueprint = blueprint_library.find("sensor.camera.rgb")
    sensor_blueprint.set_attribute("image_size_x", f"{IMG_WIDTH}")
    #sensor_blueprint.set_attribute("image_size_y", f"{IMG_HEIGHT}")
    #sensor_blueprint.set_attribute("fov", "110")
    sensor_spawn_point = carla.Transform(
        carla.Location(z = 1.5)
    )  # Posizione della telecamera sulla macchina (Relativa all'auto scelta sopra)

    try:
        time.sleep(2)
        sensor = world.spawn_actor(sensor_blueprint, sensor_spawn_point, attach_to=vehicle)
        actor_list.append(sensor)
    except RuntimeError as e:
        print("Failed to spawn sensor: {}".format(e))
    
    # sensor.listen(
    #     lambda image: save_image(image)
    # )  # image.frame può essere sostituito con l'etichetta

    # vehicle.set_autopilot(False)  # La macchina utilizzerà l'IA di base per guidare
    # while True:
    #     if keyboard.is_pressed("q"):
    #         break
    #     time.sleep(0.1)
    # time.sleep(60)
    # for i in range(60):
    #     print(vehicle.get_location())
    #     time.sleep(1)
    print("Vehicle Autopilot: OFF")
    
    time.sleep(5)
finally:  # Una volta terminato il tempo distrugge gli attori creati
    print("destroying actors")
    for actor in actor_list:
        actor.destroy()
    print("done.")
