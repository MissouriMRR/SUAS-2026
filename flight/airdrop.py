"""Functions to handle drops for the Airdrop state."""

from dronekit import Vehicle
from pymavlink.mavutil import mavlink


async def set_servo(vehicle: Vehicle, servo_num: int, pwm: int) -> None:
    """Set the PWM of an auxillary pin on the drone."""
    vehicle.message_factory.command_long_send(
        0, 0, mavlink.MAV_CMD_DO_SET_SERVO, 0, servo_num + 9, pwm, 0, 0, 0, 0, 0
    )
