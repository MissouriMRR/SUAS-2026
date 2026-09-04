import os

# Set by mavproxy to enable MAVLink 2.0
os.environ.setdefault("MAVLINK20", "1")

# MAVLink constants
GCS_SYSTEM_ID: int = 255

# mrrproxy constants
# Default system ID for the proxy's own heartbeats
DEFAULT_PROXY_SYSTEM_ID: int = 254
# Seconds to wait before reopening a link after a failure
DEFAULT_REOPEN_INTERVAL: float = 5.0
# Seconds to wait before allowing a link that recently recovered to be considered healthy
DEFAULT_RECOVER_TIME: float = 10.0
# Seconds of no msgs before link is considered down
DEFAULT_LINK_TIMEOUT: float = 5.0
# Seconds between proxy status logs
DEFAULT_STATUS_INTERVAL: float = 15.0
# Stream rate in Hz for MAVLink streams (default of 4 taken from dronekit)
DEFAULT_STREAM_RATE: int = 4
