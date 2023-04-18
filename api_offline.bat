@echo off

set SETTINGS_FILE_FOR_DYNACONF="settings-offline.toml"

set PYTHON=".\system\python\Scripts\python.exe"
set GIT=
set VENV_DIR=.\system\python

:activate_venv
set PYTHON="%VENV_DIR%\Python.exe"
echo venv %PYTHON%


:launch
%PYTHON% app_api.py %*
pause
exit /b
