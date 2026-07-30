"""Modbus-RTU indicator lamps (serial). Called by the control thread. See CLAUDE.md."""

import time


class LampBank:
    def __init__(self, enable, port, slave, refresh_secs=2.0, log=print):
        self.enable = enable
        self.port = port
        self.slave = slave
        self.refresh_secs = refresh_secs
        self.log = log
        self._client = None
        self._last = {}
        self._last_refresh = 0.0
        self._backoff = 1.0
        self._next_try = 0.0

    def apply(self, regs):
        if not self.enable:
            changed = {a: v for a, v in regs.items() if self._last.get(a) != v}
            if changed:
                self._last.update(regs)
                self.log(f"[modbus] (disabled) regs -> {sorted(regs.items())}")
            return
        if not self._connect():
            return
        now = time.time()
        refresh = self.refresh_secs and now - self._last_refresh >= self.refresh_secs
        try:
            for addr, val in regs.items():
                if refresh or self._last.get(addr) != val:
                    rr = self._client.write_register(addr, 1 if val else 0, slave=self.slave)
                    if rr is None or (hasattr(rr, "isError") and rr.isError()):
                        raise OSError(f"write_register({addr}, {val}) -> {rr}")
                    self._last[addr] = val
            if refresh:
                self._last_refresh = now
        except Exception as e:
            self.log(f"[modbus] write error: {e!r}; reconnecting")
            self._drop()

    def _connect(self):
        if self._client is not None:
            return True
        now = time.time()
        if now < self._next_try:
            return False
        try:
            from pymodbus.client import ModbusSerialClient
            c = ModbusSerialClient(port=self.port, baudrate=9600, parity="N",
                                   stopbits=1, bytesize=8, timeout=1.0)
            if not c.connect():
                raise OSError(f"cannot open {self.port}")
            self._client = c
            self._last = {}
            self._backoff = 1.0
            self.log(f"[modbus] connected {self.port} @ 9600 8N1")
            return True
        except Exception as e:
            self._next_try = now + self._backoff
            self._backoff = min(self._backoff * 2, 30.0)
            self.log(f"[modbus] connect failed: {e!r}; retry in {self._backoff:.0f}s")
            return False

    def _drop(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
        self._next_try = time.time() + self._backoff
        self._backoff = min(self._backoff * 2, 30.0)

    def close(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
