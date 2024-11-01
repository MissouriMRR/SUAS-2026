"""Fixes an error that occurs due to the dronekit version on PyPI being out of date"""

import collections
import collections.abc

collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
