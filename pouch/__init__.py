# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import threading


__version__ = "0.0.dev0"

__all__ = ["RuntimeOpts", "RUNTIME"]


class RuntimeOpts(threading.local):
    """Run-time configuration controller."""

    def __init__(self, figsave_stamp=True):
        """Initialise run-time configuration."""
        self.__dict__["figsave_stamp"] = figsave_stamp

    def __repr__(self):
        """Repr of run-time configuration."""
        keyval_pairs = [f"{key}={self.__dict__[key]}" for key in self.__dict__]
        msg = f"RuntimeOpts({', '.join(keyval_pairs)})"
        return msg

    def __setattr__(self, name, value):
        """Set attributes."""
        if name not in self.__dict__:
            msg = "'RuntimeOpts' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        self.__dict__[name] = value


RUNTIME = RuntimeOpts()
