@echo off
setlocal

set "HERE=%~dp0"
set "ARCGIS_BIN=%ProgramFiles(x86)%\ArcGIS\Desktop10.2\bin"
if not exist "%ARCGIS_BIN%\ESRI.ArcGIS.Display.dll" set "ARCGIS_BIN=%ProgramFiles%\ArcGIS\Desktop10.2\bin"

set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v3.5\csc.exe"

if not exist "%ARCGIS_BIN%\ESRI.ArcGIS.Display.dll" (
  echo ArcGIS Desktop 10.2 assemblies were not found.
  echo Expected: %ARCGIS_BIN%
  pause
  exit /b 1
)

if not exist "%CSC%" (
  echo C# compiler was not found.
  pause
  exit /b 1
)

"%CSC%" /nologo /target:exe /out:"%HERE%build_style_arcobjects_10_2.exe" ^
  /reference:"%ARCGIS_BIN%\ESRI.ArcGIS.System.dll" ^
  /reference:"%ARCGIS_BIN%\ESRI.ArcGIS.Display.dll" ^
  /reference:"%ARCGIS_BIN%\ESRI.ArcGIS.Framework.dll" ^
  /reference:"%ARCGIS_BIN%\ESRI.ArcGIS.Version.dll" ^
  "%HERE%build_style_arcobjects_10_2.cs"

if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

"%HERE%build_style_arcobjects_10_2.exe" "%HERE%MapGenAI_POI_Classic.style"
if errorlevel 1 (
  echo Style generation failed.
  pause
  exit /b 1
)

echo Created: %HERE%MapGenAI_POI_Classic.style
pause
