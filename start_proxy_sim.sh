#!/bin/bash

# Source virtual environment
source .venv/bin/activate
# Start proxy
# Primary and secondary should come from the sim or from additional mavproxy outputs
# Output: udpout:127.0.0.1:14750, udpout:127.0.0.1:14751
uv run -m proxy --primary udp:127.0.0.1:14650 --secondary udp:127.0.0.1:14651 --out udpout:127.0.0.1:14750 --out udpout:127.0.0.1:14751
