@echo off
setlocal enabledelayedexpansion

echo ==============================================
echo 👻 Silent Virtual CCTV Server (Background Mode)
echo ==============================================

REM 1. 파일 개수 자동 카운트
set "fileCount=0"
for %%f in (Video*.mp4) do (
    set /a "fileCount+=1"
)

if !fileCount! equ 0 (
    echo [ERROR] No 'Video*.mp4' files found!
    pause
    exit /b
)

echo Found !fileCount! video files.
echo Starting !fileCount! cameras in background...

REM 2. 백그라운드 송출 시작 (창 안 뜸!)
FOR /L %%i IN (1,1,!fileCount!) DO (
    if exist "Video%%i.mp4" (
        echo [STARTING] cam%%i...
        REM 💡 여기가 핵심: /B (백그라운드), > NUL 2>&1 (로그 휴지통)
        start /B "" "ffmpeg" -re -stream_loop -1 -i Video%%i.mp4 -c copy -f rtsp rtsp://localhost:8554/cam%%i > NUL 2>&1
    )
)

echo.
echo ==============================================
echo ✅ All !fileCount! cameras are running Silently!
echo ==============================================
echo [WARNING] Please DO NOT close this window.
echo Press ANY KEY to STOP all background cameras...
pause > NUL

REM 3. 깔끔한 종료 스위치
echo Stopping all cameras...
taskkill /F /IM ffmpeg.exe > NUL 2>&1
echo Done. System Safely Stopped!
pause