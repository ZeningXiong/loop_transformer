@echo off
chcp 65001 >nul
title SenseNova-Loop-9B-Preview 下载
set PYTHONIOENCODING=utf-8
"C:\Users\20495\AppData\Local\Programs\Python\Python312\python.exe" -I "%~dp0download_sensenova_loop.py"
echo.
pause
