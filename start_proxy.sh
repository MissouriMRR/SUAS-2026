#!/bin/bash

# Source virtual environment to use uv
source .venv/bin/activate

# Start proxy
# Primary (Herelink): udpout:192.168.43.1:14552
# Secondary (RFD900): /dev/ttyUSB0,57600
# Output: udpout:127.0.0.1:14750, udpout:127.0.0.1:14751
uv run -m proxy --primary udpout:192.168.43.1:14552 --secondary /dev/ttyUSB0,57600 --out udpout:127.0.0.1:14750 --out udpout:127.0.0.1:14751
