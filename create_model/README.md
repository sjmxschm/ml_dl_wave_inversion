# `create_model` description

This file describes how to use the automated simulation pipeline on the Gatech
PACE cluster. Therefore, most scripts are optimized to allow testing on a
local Windows machine as well as the final computations on a LINUX computer.

created by: Max Schmitz - mschmitz7@gatech.edu on 01032022

________________________________________________________________________________
CREATE PARAMETER SETS:
________________________________________________________________________________

Parameter sets to be simulated can be edited and created as a list in
'run_simulation.py'. The list is at the beginning of the file and called
'param_sets'. Make sure to use the right naming structure and that different
parameter sets are separated by a comma, while different parameter within a
parameter set are separated by blank spaces and a double dash, i.e. ' -- '.

________________________________________________________________________________
RUN PIPELINE ON CLUSTER:
________________________________________________________________________________

To run the simulation pipeline, copy the following files into a blank folder on
the cluster:

- create_model_script_v15.py
- extract_disp_history.py
- postprocess_2dfft_max_v15.py
- run_automated_simulations_cluster.py
- run_parallel_on_cluster.pbs
- run_parallel_simulations.py
- run_simulation.py
- utils.py

`run_parallel_simulations.py` allows to run automated parallel simulations on the cluster and is
called within the activated `wave_env_cl` python environment. This script runs directly on the end node
so make sure the following commands have been used to establish the respective environment __BEFORE__ running this script:

```bat
>> module load anaconda3/2020.11  & rem (load anaconda)
>> conda activate wave_env_cl     & rem (activate environment)
```

Navigate with the command line/bash into the corresponding folder with the files mentioned above and type
```bat
>> python3 run_parallel_simulations.py
```

to activate the pipeline. Then, for each parameter set a new folder is
created. All scripts are copied into the respective folder and a new
job is started for each parameter set. If more than one running job should
be deleted, use 'qdel_automation.py' and specify the respective job IDs.

________________________________________________________________________________