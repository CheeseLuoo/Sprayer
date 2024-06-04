REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./Sprinkler_Scheduling/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
@REM for %%s in (20 40 60 80 100 150 200 250 300) do (
for %%s in (400 500 600 700) do (    
    for %%t in (EffectOrientedSelectiveSpray) do (
        (
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 1
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 2
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 4
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 5
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --R_change_interval 3 --sourcenum 6         
            python .\main.py --config %config% --seed 0 --strategy_name %%t --team_size 2 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --team_size 3 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --team_size 4 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --team_size 5 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --team_size 6 --bound1 %%s

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