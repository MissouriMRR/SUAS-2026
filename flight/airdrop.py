"""Functions to handle drops for the Airdrop state."""

from dronekit import Vehicle
from pymavlink.mavutil import mavlink


async def set_servo(vehicle: Vehicle, servo_num: int, pwm: int) -> None:
    """
    Set the PWM of an auxillary pin on the drone.

    Parameters
    ----------
    vehicle : Vehicle
        The drone containing the servo.
    servo_num : int
        The servo instance number.
    pwm : int
        The pulse width modulation, in microseconds.
    """
    vehicle.message_factory.command_long_send(
        0, 0, mavlink.MAV_CMD_DO_SET_SERVO, 0, servo_num + 9, pwm, 0, 0, 0, 0, 0
    )
