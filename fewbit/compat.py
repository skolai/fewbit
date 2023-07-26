"""Module compat provides a few routines to manage compatibility between minor
versions of Python 3.
"""

from sys import version_info

if version_info < (3, 9):
    def removeprefix(self: str, prefix: str, /) -> str:
        if self.startswith(prefix):
            return self[len(prefix):]
        else:
            return self[:]
else:
    removeprefix = str.removeprefix
