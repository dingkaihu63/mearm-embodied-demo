@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set OPENCV_LOG_LEVEL=ERROR
set OPENCV_VIDEOIO_PRIORITY_MSMF=0

echo ============================================
echo   MeArm 工作台 v1.0
echo   实时调试与观测仪表盘
echo ============================================
echo.

python "%~dp0workbench_server.py" %*

pause
