@echo off
python -m venv env_frcnn
call env_frcnn\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pause