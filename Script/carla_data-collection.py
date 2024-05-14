"""
Script to automatically collect data from the Carla simulator

TODO: Dataset collection when the vehicle detects the signals (STOP, Speed Limit 30/60/90)
TODO: Automatic sequantial labeling of the images

"""

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

    blueprint_library = (
        world.get_blueprint_library()
    )  # List of all the available blueprint of Carla

    vehicle_blueprint = blueprint_library.find(
        "vehicle.tesla.model3"
    )  # Vehicle model to be used in the simulation

    vehicle_spawn_point = carla.Transform(
        carla.Location(x=210.86701172, y=199.44183594, z=1)
    )  # Initial position of the vehicle (Same as the camera position in the simulation)

    vehicle = world.spawn_actor(vehicle_blueprint, vehicle_spawn_point)
    actor_list.append(vehicle)

    sensor_blueprint = blueprint_library.find("sensor.camera.rgb")
    sensor_blueprint.set_attribute("image_size_x", f"{IMG_WIDTH}")
    sensor_blueprint.set_attribute("image_size_y", f"{IMG_HEIGHT}")
    sensor_blueprint.set_attribute("fov", "110")
    sensor_spawn_point = carla.Transform(
        carla.Location(x=0.5, z=1.5)
    )  # Relative Position => The position is based on the vehicle's position and model

    try:
        time.sleep(2)
        sensor = world.spawn_actor(
            sensor_blueprint, sensor_spawn_point, attach_to=vehicle
        )
        actor_list.append(sensor)
    except RuntimeError as e:
        print("Failed to spawn sensor: {}".format(e))

    sensor.listen(lambda image: save_image(image))

    # vehicle.set_autopilot(False)

    time.sleep(5)  # Execution time of the script

finally:  # After the time has passed, all the actors spawned with the scripts will be destroyed
    print("Destroying actors...")
    for actor in actor_list:
        actor.destroy()
    print("Done.")
