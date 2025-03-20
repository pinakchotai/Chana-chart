@echo off
echo Setting up Chickpeas Spot Price Chart application...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed. Please install Python 3.7 or higher.
    exit /b 1
)

for /f "tokens=*" %%a in ('python --version') do set python_version=%%a
echo Python found: %python_version%

REM Check if Excel file exists
if exist "CHANA R&D.xlsx" (
    echo Excel file found.
) else (
    echo Warning: Excel file 'CHANA R&D.xlsx' not found in the current directory.
    echo Please make sure to place the Excel file in this directory.
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo To run the application:
echo 1. Activate the virtual environment if not already activated:
echo    venv\Scripts\activate
echo 2. Run the application:
echo    python app.py
echo 3. Open your browser and go to: http://localhost:8050
echo.

pause 