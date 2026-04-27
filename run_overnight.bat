@echo off
REM ============================================================
REM   Overnight v2 training pipeline
REM     - Skips shadow-free generation if already complete
REM     - Trains PyramidalNet with cosine LR + warmup,
REM       RandomResizedCrop, combined val (easy + hard),
REM       max 100 epochs with patience 15
REM     - Saves to best_pyramidal_net_shadowfree_v2.pth
REM   Logs to overnight.log
REM ============================================================

cd /d E:\TOD

set LOG=overnight.log
echo. > %LOG%
echo ============================================== >> %LOG%
echo Started at %DATE% %TIME%                       >> %LOG%
echo PyramidalNet v2 training                       >> %LOG%
echo ============================================== >> %LOG%

echo [%TIME%] Checking shadow-free dataset...
echo [%TIME%] Checking shadow-free dataset... >> %LOG%

set TRAIN_DONE=0
for /f %%A in ('dir /a-d /b "Dataset_shadow_free\train\images\*" 2^>nul ^| find /c /v ""') do set TRAIN_DONE=%%A
set VAL_DONE=0
for /f %%A in ('dir /a-d /b "Dataset_shadow_free\validation\images\*" 2^>nul ^| find /c /v ""') do set VAL_DONE=%%A
set VAL_HARD_DONE=0
for /f %%A in ('dir /a-d /b "Dataset_shadow_free\validation_hard\images\*" 2^>nul ^| find /c /v ""') do set VAL_HARD_DONE=%%A

echo Train: %TRAIN_DONE%/5000 ^| Val easy: %VAL_DONE%/788 ^| Val hard: %VAL_HARD_DONE%/212
echo Train: %TRAIN_DONE%/5000 ^| Val easy: %VAL_DONE%/788 ^| Val hard: %VAL_HARD_DONE%/212 >> %LOG%

if %TRAIN_DONE% LSS 5000 (
    echo [%TIME%] Running shadow-free generation...
    echo [%TIME%] Running shadow-free generation... >> %LOG%
    python inference_shadow_removal.py >> %LOG% 2>&1
) else if %VAL_DONE% LSS 788 (
    echo [%TIME%] Running shadow-free generation...
    echo [%TIME%] Running shadow-free generation... >> %LOG%
    python inference_shadow_removal.py >> %LOG% 2>&1
) else (
    echo [%TIME%] Shadow-free dataset complete. Skipping.
    echo [%TIME%] Shadow-free dataset complete. Skipping. >> %LOG%
)

echo [%TIME%] Starting PyramidalNet v2 training (~5-7 hours)...
echo [%TIME%] Starting PyramidalNet v2 training... >> %LOG%
echo. >> %LOG%

python PyramidalNet_train.py >> %LOG% 2>&1

echo.
echo [%TIME%] Done. See overnight.log for full output.
echo [%TIME%] Done. >> %LOG%
echo Best checkpoint: trained_models\best_pyramidal_net_shadowfree_v2.pth >> %LOG%
echo Training curves: trained_models\best_pyramidal_net_shadowfree_v2_curves.png >> %LOG%

pause
