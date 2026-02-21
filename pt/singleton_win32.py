# singleton_win32.py
# Windows-only single-instance guard using a named mutex.
# Standard library only.

from __future__ import annotations
import ctypes
from ctypes import wintypes

_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

CreateMutexW = _KERNEL32.CreateMutexW
CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
CreateMutexW.restype = wintypes.HANDLE

GetLastError = _KERNEL32.GetLastError
GetLastError.argtypes = ()
GetLastError.restype = wintypes.DWORD

ReleaseMutex = _KERNEL32.ReleaseMutex
ReleaseMutex.argtypes = (wintypes.HANDLE,)
ReleaseMutex.restype = wintypes.BOOL

CloseHandle = _KERNEL32.CloseHandle
CloseHandle.argtypes = (wintypes.HANDLE,)
CloseHandle.restype = wintypes.BOOL

ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    def __init__(self, name: str):
        self.name = name
        self.handle = None

    def acquire(self) -> bool:
        # "Global\" allows cross-session uniqueness; OK for single-user bots.
        mutex_name = f"Global\\{self.name}"
        h = CreateMutexW(None, True, mutex_name)
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")

        self.handle = h
        already = (GetLastError() == ERROR_ALREADY_EXISTS)
        return (not already)

    def release(self) -> None:
        if self.handle:
            try:
                ReleaseMutex(self.handle)
            except Exception:
                pass
            try:
                CloseHandle(self.handle)
            except Exception:
                pass
            self.handle = None
