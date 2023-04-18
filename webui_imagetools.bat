@echo off

set PYTHON=".\system\python\Scripts\python.exe"
set GIT=
set VENV_DIR=.\system\python

:activate_venv
set PYTHON="%VENV_DIR%\Python.exe"
echo venv %PYTHON%


:launch
%PYTHON% app_imagetools.py  --port 7899  %*
pause
exit /b
