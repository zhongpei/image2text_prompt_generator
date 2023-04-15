@echo off

set PYTHON="venv/scripts/python.exe"
set GIT=
set VENV_DIR=venv

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%


:launch
%PYTHON% app.py %*
pause
exit /b
