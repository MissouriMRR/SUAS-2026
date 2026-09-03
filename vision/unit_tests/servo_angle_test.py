from flight.camera import CameraIRL

if __name__ == "main":
    camera = CameraIRL()

    while True:
        gimbal_attitude = camera.camera.getAttitude()
        print(gimbal_attitude)
