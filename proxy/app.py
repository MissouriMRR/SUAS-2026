from __future__ import annotations

import logging
import signal
from typing import Annotated

import typer
from pymavlink import mavutil

from proxy import (
    DEFAULT_PROXY_SYSTEM_ID,
    DEFAULT_RECOVER_TIME,
    DEFAULT_REOPEN_INTERVAL,
    DEFAULT_STATUS_INTERVAL,
    DEFAULT_STREAM_RATE,
)
from proxy.link import Link
from proxy.proxy import MRRProxy

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command()
def main(
    primary: Annotated[
        str,
        typer.Option(help="preferred vehicle link"),
    ],
    secondary: Annotated[
        list[str] | None,
        typer.Option(
            "--secondary",
            help="fallback vehicle link, may be given more than once, in priority order",
        ),
    ] = None,
    out: Annotated[
        list[str] | None,
        typer.Option(
            help="available output links (default udpout:127.0.0.1:14550), can have multiple",
        ),
    ] = None,
    recover_time: Annotated[
        float,
        typer.Option(
            help="seconds a higher priority link must be healthy before switching back",
        ),
    ] = DEFAULT_RECOVER_TIME,
    stream_rate: Annotated[
        int,
        typer.Option(
            help="rate in Hz to request all data streams on each link",
        ),
    ] = DEFAULT_STREAM_RATE,
    source_system: Annotated[
        int,
        typer.Option(help="MAVLink system ID for the proxy's own heartbeats"),
    ] = DEFAULT_PROXY_SYSTEM_ID,
    reopen_interval: Annotated[
        float,
        typer.Option(help="seconds between attempts to reopen a failed link"),
    ] = DEFAULT_REOPEN_INTERVAL,
    status_interval: Annotated[
        float,
        typer.Option(help="seconds between status log lines, 0 to disable"),
    ] = DEFAULT_STATUS_INTERVAL,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="enable debug logging"),
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if secondary is None:
        secondary = []

    # closest thing to a proxy
    component = mavutil.mavlink.MAV_COMP_ID_UDP_BRIDGE

    devices: list[str] = [primary, *secondary]
    inputs = [
        Link(device, source_system, component, reopen_interval) for device in devices
    ]
    outputs = [
        Link(device, source_system, component)
        for device in out or ["udpout:127.0.0.1:14550"]
    ]

    proxy = MRRProxy(
        inputs,
        outputs,
        recover_time=recover_time,
        stream_rate=stream_rate,
        status_interval=status_interval,
    )
    # Stop proxy safely if interrupted
    signal.signal(signal.SIGINT, proxy.stop)
    signal.signal(signal.SIGTERM, proxy.stop)
    logger.info(
        "inputs: %s; outputs: %s",
        ", ".join(input.device for input in inputs),
        ", ".join(output.device for output in outputs),
    )
    proxy.run()


if __name__ == "__main__":
    app()
