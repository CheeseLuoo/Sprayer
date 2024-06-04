REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./Sprinkler_Scheduling/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
@REM for %%s in (7 11 18 20 25 36 42 50 60 72 80 85) do (
for %%s in (1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 ) do (
    for %%t in (EffectOrientedMCTSSpray) do (
        (
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 8 --adaptive_step 8 --bound1 %%s
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 2
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 4
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 5
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 6         
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 3
        ) || (
            REM Append the ERRORLEVEL (PID) to the pids variable
            set "pids=!pids!!ERRORLEVEL!!"
            echo Terminating processes...
            REM Terminate all background processes
            for %%p in (%pids%) do (
                taskkill /PID %%p /F >nul
            )
            exit /b 1
        )
    )
)

exit /b 1