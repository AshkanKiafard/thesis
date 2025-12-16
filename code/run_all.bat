python finetune.py || exit /b
python pre_embed.py || exit /b
python evaluation.py || exit /b
pause