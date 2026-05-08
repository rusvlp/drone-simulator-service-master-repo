@echo off
setlocal

echo [1/5] Pulling main repository...
git pull
if %ERRORLEVEL% neq 0 (
    echo ERROR: git pull failed
    exit /b %ERRORLEVEL%
)

echo [2/5] Initialising submodules...
git submodule update --init --recursive
if %ERRORLEVEL% neq 0 (
    echo ERROR: submodule init failed
    exit /b %ERRORLEVEL%
)

echo [3/5] diploma-course-service ^(docker-integration^)...
cd services\diploma-course-service
git fetch origin
git checkout docker-integration
git pull origin docker-integration
cd ..\..
if %ERRORLEVEL% neq 0 ( echo ERROR: diploma-course-service & exit /b %ERRORLEVEL% )

echo [4/5] diploma-frontend ^(docker-integration^)...
cd services\diploma-frontend
git fetch origin
git checkout docker-integration
git pull origin docker-integration
cd ..\..
if %ERRORLEVEL% neq 0 ( echo ERROR: diploma-frontend & exit /b %ERRORLEVEL% )

echo [5/5] diploma-gateway ^(docker-integraion^)...
cd services\diploma-gateway
git fetch origin
git checkout docker-integraion
git pull origin docker-integraion
cd ..\..
if %ERRORLEVEL% neq 0 ( echo ERROR: diploma-gateway & exit /b %ERRORLEVEL% )

echo [5/5] diploma-user-service ^(master^)...
cd services\diploma-user-service
git fetch origin
git checkout master
git pull origin master
cd ..\..
if %ERRORLEVEL% neq 0 ( echo ERROR: diploma-user-service & exit /b %ERRORLEVEL% )

echo Done.
endlocal
