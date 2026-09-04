from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import cast

from pymavlink import mavutil
from pymavlink.dialects.v20.all import MAVLink_message

from proxy import DEFAULT_LINK_TIMEOUT, DEFAULT_REOPEN_INTERVAL

logger = logging.getLogger(__name__)


class Link:
    """Represents a MAVLink connection that reopens itself after any failure.

    Attributes
    ----------
    device : str
        pymavlink/mavproxy connection string, could contain baud rate using a comma
    conn : mavutil.mavtcp | mavutil.mavtcpin | mavutil.mavudp | mavutil.mavserial | None
        The underlying pymavlink connection, or None while the link is closed.
    last_rx : float
        Monotonic time of the last message received from the vehicle.
    alive_since : float
        Monotonic time at which the link most recently became healthy.
    """

    def __init__(
        self,
        device: str,
        source_system: int,
        source_component: int,
        reopen_interval: float = DEFAULT_REOPEN_INTERVAL,
        alive_timeout: float = DEFAULT_LINK_TIMEOUT,
    ) -> None:
        device, _, baud = device.partition(",")
        self.baud: int = int(baud) if baud else 57600

        self.device: str = device
        self.source_system: int = source_system
        self.source_component: int = source_component
        self.reopen_interval: float = reopen_interval
        self.alive_timeout: float = alive_timeout

        self.conn: (
            mavutil.mavtcp
            | mavutil.mavtcpin
            | mavutil.mavudp
            | mavutil.mavserial
            | None
        ) = None

        self.last_rx: float = 0.0
        self.alive_since: float = 0.0
        self.up: bool = False
        self.streams_requested: set[tuple[int, int]] = set()

        self._next_open: float = 0.0

    def _open_mavlink_conn(
        self, device: str, baud: int, source_system: int, source_component: int
    ):
        """
        Equivalent to `mavutil.mavlink_connection`, only contains connection types
        that we care about, makes typing simpler.
        """
        if device.startswith("tcp:"):
            return mavutil.mavtcp(
                device[4:],
                autoreconnect=True,
                source_system=source_system,
                source_component=source_component,
            )
        if device.startswith("tcpin:"):
            return mavutil.mavtcpin(
                device[6:],
                source_system=source_system,
                source_component=source_component,
            )
        if device.startswith(("udpin:", "udpout:", "udp:")):
            return mavutil.mavudp(
                device.split(":", 1)[1],
                input=device.startswith(("udpin:", "udp:")),
                source_system=source_system,
                source_component=source_component,
            )
        else:
            return mavutil.mavserial(
                device,
                baud=baud,
                source_system=source_system,
                source_component=source_component,
                autoreconnect=True,
            )

    def open(self) -> bool:
        """Try to open the connection if it is closed and the retry timer allows it.

        Returns
        -------
        bool
            Whether the connection was successfully opened.
        """
        # If the link is already open no need to reopen
        if self.conn is not None:
            return True

        now = time.monotonic()
        if now < self._next_open:
            # Connection is not yet ready, wait for the reopen interval
            # to check again
            return False
        self._next_open = now + self.reopen_interval

        try:
            self.conn = self._open_mavlink_conn(
                self.device,
                self.baud,
                self.source_system,
                self.source_component,
            )
        except Exception as exc:
            logger.warning("%s: cannot open (%s), retrying", self.device, exc)
            self.conn = None
            return False

        self.streams_requested.clear()
        logger.info("%s: opened", self.device)
        return True

    def close(self) -> None:
        """Close connection, if open."""
        if self.conn is None:
            return
        try:
            self.conn.close()
        except Exception as exc:
            logger.warning("%s: error while closing (%s)", self.device, exc)
        self.conn = None
        self.last_rx = 0.0
        self.alive_since = 0.0

    def fd(self) -> int | None:
        """
        Return the file descriptor for the device,
        or None if not available.
        """
        if self.conn is None:
            return None
        fd = self.conn.fd
        if fd is None:
            return None
        return fd if fd >= 0 else None

    def read(self) -> Iterator[MAVLink_message]:
        """
        Read all messages currently available on the link.
        This is used as an iterator so that other code just do
        "for msg in read" and will automatically stop reading
        when there is no msg to read.
        """
        if self.conn is None:
            return
        while True:
            try:
                # Need to cast here because base mavfile returns Never
                # bc the recv function is not implemented
                msg: MAVLink_message | None = cast(
                    MAVLink_message | None, self.conn.recv_msg()
                )
            except Exception as exc:
                logger.warning("%s: read error (%s), reopening", self.device, exc)
                self.close()
                # Return to end iterator
                return
            if msg is None:
                # Return to end iterator
                return
            msg_type: str = msg.get_type()
            if msg_type == "BAD_DATA":
                continue
            # Provide msg to iterator
            yield msg

    def write(self, buf: bytes | bytearray) -> None:
        """
        Just an error-catch around mavfile.write.

        Parameters
        ----------
        buf : bytes | bytearray
            MAVLink frame/message to write.
        """
        if self.conn is None:
            return
        try:
            self.conn.write(buf)
        except Exception as exc:
            logger.warning("%s: write error (%s), reopening", self.device, exc)
            self.close()

    def alive(self) -> bool:
        """
        Whether vehicle traffic has been seen on this link
        within the last `alive_timeout` seconds.
        """
        now = time.monotonic()
        return (
            self.conn is not None
            and self.last_rx > 0.0
            and now - self.last_rx < self.alive_timeout
        )

    def alive_for(self, now: float) -> float:
        """Seconds the link has been continuously healthy."""
        return now - self.alive_since if self.up else 0.0

    def update_last_rx(self) -> None:
        """Update last_rx to reflect the arrival of a vehicle message."""
        now = time.monotonic()
        if not self.alive():
            self.alive_since = now
        self.last_rx = now

    def refresh_state(self) -> None:
        """Check if the link is still alive and update `up` accordingly."""
        is_alive = self.alive()
        if is_alive and not self.up:
            logger.info("%s: link up", self.device)
        elif not is_alive and self.up:
            logger.warning(
                "%s: link down (no vehicle traffic for %.1fs)",
                self.device,
                self.alive_timeout,
            )
        self.up = is_alive
