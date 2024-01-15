# Benchmark Test Case 02: Amplitude Estimation

This part of the repository is associated with the Benchmark Test Case (BTC) 02: **Amplitude Estimation**. 

For executing a complete BTC fits thing you need to select the Amplitude Estimation algorithm that you want to use. This can be done in the variable **AE** at the end part of the **my_benchmark_execution.py** file. Once the algorithm is set you need to configure the Amplitude Estimation algorithm. This is done by editing the different **json** files fo the **jsons** folder. Valid strings for  **AE** variable are:

* MCAE (MonteCarlo Amplitude Estimation algorithm). Configuration json is: integral_mcae_configuration.json 
* CQPEAE (Classical Quantum Phase Estimation Amplitude Estimation algorithm). Configuration json is: integral_cqpeae_configuration.json 
* IQPEAE (Iterative Quantum Phase Estimation Amplitude Estimation algorithm). Configuration json is: integral_iqpeae_configuration.json 
* MLAE (Maximum Likelihood Amplitude Estimation algorithm). Configuration json is: integral_mlae_configuration.json 
* IQAE (Iterative Quantum Amplitude Estimation algorithm). Configuration json is: integral_iqae_configuration.json 
* RQAE (Real Qauantum Amplitude Estimation algorithm). Configuration json is: integral_rqae_configuration.json 

Additionally the list with the number of qubits can be provided by editing the key: *list_of_qbits* of the *benchmark_arguments* dictionary. 

For executing a complete BTC you need to configure the **kernel_configuration** dictionary in final part of the **my_benchmark_execution.py** file. The files will be saved in the **Results** folder (you can change the folder in key argument **saving_folder** of the **benchmark_arguments** dictionary). Then you should execute the following command line:

    python my_benchmark_execution.py

If you want a complete execution and create the complete **JSON** file with the complete information of the benchmark results following the **NEASQC JSON Schema** you can instead execute the script **benchmark_exe.sh**:
 
    bash benchmark_exe.sh

All the results files and the corresponding JSON will be stored in the **Results** folder.

