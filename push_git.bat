@echo off
setlocal enabledelayedexpansion

set "MSG=%~1"
if "%MSG%"=="" set "MSG=update"

echo ================================================
echo  Git commit + push
echo  Message: %MSG%
echo ================================================

:: ── helper: commit and push in current dir ───────────────────
:: usage: call :push_sub <branch>
goto :main

:push_sub
    git checkout %1 2>nul
    git add -A
    git diff --cached --exit-code --quiet 2>nul
    if %ERRORLEVEL% neq 0 (
        git commit -m "%MSG%"
        git push origin %1
        if %ERRORLEVEL% neq 0 ( echo ERROR pushing %CD% & exit /b 1 )
    ) else (
        echo   nothing to commit
    )
    exit /b 0

:main

:: ── diploma-course-service ──────────────────────────────────
echo.
echo [1/5] diploma-course-service  (docker-integration)
cd services\diploma-course-service
call :push_sub docker-integration
cd ..\..

:: ── diploma-frontend ────────────────────────────────────────
echo.
echo [2/5] diploma-frontend  (docker-integration)
cd services\diploma-frontend
call :push_sub docker-integration
cd ..\..

:: ── diploma-gateway ─────────────────────────────────────────
echo.
echo [3/5] diploma-gateway  (docker-integraion)
cd services\diploma-gateway
call :push_sub docker-integraion
cd ..\..

:: ── diploma-user-service ────────────────────────────────────
echo.
echo [4/5] diploma-user-service  (master)
cd services\diploma-user-service
call :push_sub master
cd ..\..

:: ── master repo (includes terrain-service, terrain-gen, etc.) ──
echo.
echo [5/5] master repo  (master)
git add -A
git diff --cached --exit-code --quiet 2>nul
if %ERRORLEVEL% neq 0 (
    git commit -m "%MSG%"
    git push origin master
    if %ERRORLEVEL% neq 0 ( echo ERROR pushing master repo & exit /b 1 )
) else (
    echo   nothing to commit
)

echo.
echo ================================================
echo  Done.
echo ================================================
endlocal
