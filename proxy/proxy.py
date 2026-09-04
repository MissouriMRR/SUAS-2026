from __future__ import annotations

import logging
import select
import time

from pymavlink import mavutil
from pymavlink.dialects.v20.all import MAVLink_message

from proxy import (
    DEFAULT_RECOVER_TIME,
    DEFAULT_STATUS_INTERVAL,
    DEFAULT_STREAM_RATE,
    GCS_SYSTEM_ID,
)
from proxy.link import Link

logger = logging.getLogger(__name__)


class MRRProxy:
    """
    Combines multiple vehicle links into a central link, where it acts
    as a proxy where the primary link is forwarded when up but other feeds
    can act as fallbacks.

    Parameters
    ----------
    inputs : list[Link]
        Link to use as inputs.
    outputs : list[Link]
        Links that input messages are forwarded to.
    recover_time : float
        Time to wait before switching to a fallback link.
    stream_rate : int
        Rate at which to request stream data from vehicles.
    status_interval : float
        Interval at which to log status information.
    """

    def __init__(
        self,
        inputs: list[Link],
        outputs: list[Link],
        recover_time: float = DEFAULT_RECOVER_TIME,
        stream_rate: int = DEFAULT_STREAM_RATE,
        status_interval: float = DEFAULT_STATUS_INTERVAL,
    ) -> None:
        self.inputs: list[Link] = inputs
        self.outputs: list[Link] = outputs
        self.recover_time: float = recover_time
        self.stream_rate: int = stream_rate
        self.status_interval: float = status_interval

        self.active: Link | None = None
        self.vehicles: set[tuple[int, int]] = set()
        self.running: bool = True

        self._last_heartbeat: float = 0.0
        self._last_status: float = 0.0

    def stop(self, *_: object) -> None:
        """
        Stop all links. *arg is needed to catch
        additional arguments from signal handlers.
        """
        self.running = False

    def run(self) -> None:
        """
        Run proxy steps until the proxy is stopped,
        clean up all links after proxy is stopped.
        """
        try:
            while self.running:
                self._step()
        finally:
            for link in self.inputs + self.outputs:
                link.close()
            logger.info("Proxy stopped.")

    def _step(self) -> None:
        """
        Handles one "step" of the proxy, where all links are checked for
        open/ready state and all messages are read/written as necessary.
        """
        # Get all open links
        # Will also attempt to open any closed links in inputs/outputs
        open_links = [link for link in self.inputs + self.outputs if link.open()]

        # Wait for any open links to be ready for reading
        # by checking/grabbing their file descriptors
        fds: list[int] = []
        for link in open_links:
            fd = link.fd()
            if fd is not None:
                fds.append(fd)
        if fds:
            try:
                # Wait for fd to become ready for reading
                # Timeout after 0.05s
                select.select(fds, [], [], 0.05)
            except (OSError, ValueError):
                # Retry in next step
                pass
        else:
            # to prevent looping too quickly
            time.sleep(0.05)

        now = time.monotonic()
        for input in self.inputs:
            # Read/handle all messages from the vehicle links
            # even if it isn't the primary one
            for msg in input.read():
                self._handle_input_message(input, msg)
        for output in self.outputs:
            # Read/handle all messages from the client links
            for msg in output.read():
                self._handle_output_message(msg)

        # Refresh link state and update active link
        for link in self.inputs:
            link.refresh_state()
        self._update_active(now)

        # Send heartbeats/status logs if needed
        if now - self._last_heartbeat >= 1.0:
            self._last_heartbeat = now
            self._send_heartbeats()
        if self.status_interval > 0 and now - self._last_status >= self.status_interval:
            self._last_status = now
            self._log_status(now)

    def _handle_input_message(self, link: Link, msg: MAVLink_message) -> None:
        """
        Parses messages from input Links to determine whether it is a message from the vehicle
        itself, and handles it accordingly. If the message is from the active Link then it is
        forwarded to all outputs Links.
        """
        source: tuple[int, int] = (msg.get_srcSystem(), msg.get_srcComponent())
        # Need to make sure that we only forward vehicle messages and not
        # GCS messages like that ones that MissionPlanner sends
        from_vehicle = source[0] not in (
            GCS_SYSTEM_ID,
            link.source_system,
        )

        if from_vehicle:
            link.update_last_rx()
            # HEARTBEAT msgs can be used to detect vehicle presence
            # https://mavlink.io/en/services/heartbeat.html
            # this is done in dronekit/pymavlink as well
            if msg.get_type() == "HEARTBEAT":
                self.vehicles.add(source)
                if source not in link.streams_requested:
                    link.streams_requested.add(source)
                    self._request_streams(link, source)
        if link is self.active:
            # If the link is the active one then we want
            # to forward the message to all outputs
            buf = msg.get_msgbuf()
            for output in self.outputs:
                output.write(buf)

    def _handle_output_message(self, msg: MAVLink_message) -> None:
        """
        Take messages coming back from the output Links and
        forward them to the active input Link.
        """
        if self.active is not None:
            self.active.write(msg.get_msgbuf())

    def _update_active(self, now: float) -> None:
        """
        Check if active link needs to be updated to a new link,
        and if so, request streams for all vehicles in the new link.
        """
        chosen = self._choose_active(now)
        if chosen is self.active:
            # The link we want is already the active one
            return

        previous = self.active
        self.active = chosen

        # Log link change
        if chosen is None:
            logger.error("no links available, switching to no active link")
            return
        if previous is not None:
            logger.warning(
                "switching feed from %s to %s",
                previous.device,
                chosen.device,
            )
        else:
            logger.warning("switching feed to %s", chosen.device)

        # Request streams for all vehicles in new link
        for vehicle in self.vehicles:
            self._request_streams(chosen, vehicle)

    def _choose_active(self, now: float) -> Link | None:
        """
        Pick the link to use as the active input Link based on
        priority (order provided by input Link order) and whether
        the links are alive or not.
        """
        current_link = self.active
        current_link_alive = current_link is not None and current_link.alive()

        # Check links in order of priority (list order)
        for link in self.inputs:
            # If the link is the current link and its alive, keep it active
            if link is current_link and current_link_alive:
                return current_link

            if not link.alive():
                continue
            if not current_link_alive:
                # The current link is not alive and this one is,
                # use this one
                return link
            if link.alive_for(now) >= self.recover_time:
                # Both the current link and this link are alive,
                # but this link is higher priority and has been alive
                # for long enough, use it
                return link

        # No active link, return the primary link as long as it's open
        # even if it's down it is good to send commands as soon as it comes back
        if current_link is None:
            return (
                self.inputs[0]
                if self.inputs and self.inputs[0].conn is not None
                else None
            )
        return current_link

    def _request_streams(self, link: Link, vehicle: tuple[int, int]) -> None:
        """
        In order to request all messages from a vehicle, you actually need to send a
        `request_data_stream` message with `MAV_DATA_STREAM_ALL`, as without it the vehicle
        will only send a default set of messages. Dronekit also does this when connecting to
        vehicles, so the same settings are used to ensure all messages that are needed
        are received.
        This request is sent to all vehicles that are detected on the given link.

        This is also done in mavproxy under `set_stream_rates`.

        Some more info on this:
            https://mavlink.io/en/mavgen_python/howto_requestmessages.html
            https://ardupilot.org/dev/docs/mavlink-requesting-data.html#using-request-data-stream
        """
        if link.conn is None:
            return
        try:
            link.conn.mav.request_data_stream_send(
                vehicle[0],
                vehicle[1],
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                self.stream_rate,
                1,
            )
        except Exception as exc:
            logger.warning("%s: failed to request streams (%s)", link.device, exc)

    def _send_heartbeats(self) -> None:
        """
        Send HEARTBEAT messages to all input feeds to keep them connected/alive.
        """
        for link in self.inputs:
            if link.conn is None:
                continue
            try:
                link.conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    0,
                )
            except Exception as exc:
                logger.warning("%s: HEARTBEAT failed (%s), reopening", link.device, exc)
                link.close()

    def _log_status(self, now: float) -> None:
        """
        Print the status of all input links, including their state and last received time.
        """
        parts: list[str] = []
        for link in self.inputs:
            state = "closed" if link.conn is None else ("up" if link.up else "down")
            marker = ">" if link is self.active else " "
            age = now - link.last_rx if link.last_rx > 0 else float("inf")
            parts.append(f"{marker}{link.device}={state} last={age:.1f}s")
        logger.info("status:\n%s", "\n".join(parts))
