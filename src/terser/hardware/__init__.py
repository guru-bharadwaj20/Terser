"""Hardware abstraction: resolve a hardware config into a concrete device and precision."""

from terser.hardware.device import DeviceInfo, HardwareUnavailableError, get_device

__all__ = ["DeviceInfo", "HardwareUnavailableError", "get_device"]
