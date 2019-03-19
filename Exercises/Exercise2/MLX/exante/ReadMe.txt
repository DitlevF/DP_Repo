*******************************************************************************************
* 1. Functions for solving, simulating, and estimating the buffer-stock consumption model *
*******************************************************************************************

m-files:
funs.m: basic functions for finding GaussHermite nodes, printing figures etc.
model.m: functions for setting up model, solving it and simulating from it
estimate.m: functions for estimating the model
figs.m: function for plotting the solutions and simulations

Notebooks:
run_01_perfect_foresight_con.mlx: solve and simulate the perfect forsight model with a borrowing constraint
run_02_buffer_stock.mlx: solve and simulate the buffer-stock model in infinite horizon
run_03_compare_preferences:compare solutions across preferences
run_04_lifecycle.mlx: solve and simulate the buffer-stock model with a life-cycle

figs/: folder for all produced figures


********************************************************************************************
* 2. Functions for solving a consumption-saving model with a discrete absorbing retirement *
********************************************************************************************

m-files:
funs.m: basic functions for finding GaussHermite nodes, printing figures etc.
model_dc.m: functions for setting up the model and solving it
figs_dc.m: function for plotting the solutions

run_05_dc.mlx: solve the model

figs/: folder for all produced figures


**************************
* 3. Is MATLAB too slow? *
**************************

Save the .mlx files as .m files:
1. open the .mlx file in the Live Editor
2. in the Live Editor tab choose 'Save' -> 'Save as'
3. save as a .m file
