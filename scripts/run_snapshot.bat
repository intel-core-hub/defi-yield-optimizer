@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" "src\snapshot.py" >> "data\snapshot.log" 2>&1
