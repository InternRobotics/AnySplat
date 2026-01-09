@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_HOME%\bin;%PATH%"
cd /d D:\Phone3dgs_v2\AnySplat
call anysplat_env\Scripts\activate.bat
python demo_gradio.py
