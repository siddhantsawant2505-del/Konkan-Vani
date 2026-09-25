@echo off
cd /d "%~dp0"
py -X utf8 src\build_index.py > data\processed\index_build.log 2>&1
echo DONE > data\processed\index_build_done.flag
