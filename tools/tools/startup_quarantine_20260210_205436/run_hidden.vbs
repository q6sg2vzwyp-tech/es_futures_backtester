' run_hidden.vbs
' Runs a command completely hidden (no console window)
Option Explicit
Dim shell, cmd
Set shell = CreateObject("WScript.Shell")
cmd = WScript.Arguments(0)
shell.Run cmd, 0, False
