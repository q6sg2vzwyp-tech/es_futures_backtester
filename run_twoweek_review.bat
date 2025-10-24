@echo off
setlocal
cd /d "%~dp0"
set PY=C:\Users\owner\AppData\Local\Programs\Python\Python313\python.exe
"%PY%" twoweek_review.py 1>twoweek_review.out 2>twoweek_review.err
echo.
echo ===== twoweek_review.py finished =====
for %%F in (twoweek_review.err) do if exist "%%F" if %%~zF GTR 0 (
  echo ----- ERRORS -----
  type "twoweek_review.err"
)
echo Output saved to twoweek_review.out
pause
