@echo off
rem Thin shim so the bridge works regardless of which python is first on PATH.
setlocal
set "PY=%USERPROFILE%\documents-toolkit\.venv\Scripts\python.exe"
if exist "%PY%" goto run
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
:run
%PY% "%~dp0colab.py" %*
