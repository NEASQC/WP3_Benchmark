# Benchmark Test Case 03: Quantum Phase Estimartion

This part of the repository is associated with the Benchmark Test Case (BTC) 03: **Quantum Phase Estimation** (QPE). 

For executing a complete BTC you need to configure the **kernel_configuration** dictionary in final part of the **my_benchmark_execution.py** file. The files will be saved in the **Results** folder (you can change the folder in key argument **saving_folder** of the **benchmark_arguments** dictionary). Then you should execute the following command line:

    python my_benchmark_execution.py

If you want a complete execution and create the complete **JSON** file with the complete information of the benchmark results following the **NEASQC JSON Schema** you can instead execute the script **benchmark_exe.sh**:
 
    bash benchmark_exe.sh

All the results files and the corresponding JSON will be stored in the **Results** folder.

