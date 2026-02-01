@echo off
REM Granite 4.0-1B 微调工具 - Windows启动脚本
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM 检查conda环境
call conda activate granite_finetune 2>nul
if errorlevel 1 (
    echo [提示] 未找到granite_finetune环境，请先运行: conda env create -f environment.yml
    exit /b 1
)

REM 执行命令
python run.py %*
