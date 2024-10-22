BTC\_02\_AE Amplitude Estimation
================================

This part of the repository is associated with the Benchmark Test Case (BTC) 02: **Amplitude Estimation**. 

Select and configure the **Amplitude Estimation** algorithm using the JSON files inside the jsons folder. Configure the **kernel_configuration** dictionary in final part of the **my_benchmark_execution.py** file. The files will be saved in the **Results** folder (you can change the folder in key argument **saving_folder** of the **benchmark_arguments** dictionary). Then you should execute the following command line:

    python my_benchmark_execution.py

If you want a complete execution and create the complete **JSON** file with the complete information of the benchmark results following the **NEASQC JSON Schema** you can instead execute the script **benchmark_exe.sh**:
 
    bash benchmark_exe.sh

All the results files and the corresponding JSON will be stored in the **Results** folder.


my\_benchmark\_execution
------------------------

.. automodule:: tnbs.BTC_02_AE.my_benchmark_execution
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_info
-------------------

.. automodule:: tnbs.BTC_02_AE.my_benchmark_info
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_summary
----------------------

.. automodule:: tnbs.BTC_02_AE.my_benchmark_summary
   :members:
   :undoc-members:
   :show-inheritance:

my\_environment\_info
---------------------

.. automodule:: tnbs.BTC_02_AE.my_environment_info
   :members:
   :undoc-members:
   :show-inheritance:

neasqc\_benchmark
-----------------

.. automodule:: tnbs.BTC_02_AE.neasqc_benchmark
   :members:
   :undoc-members:
   :show-inheritance:

ae\_sine\_integral
------------------

.. automodule:: tnbs.BTC_02_AE.ae_sine_integral
   :members:
   :undoc-members:
   :show-inheritance:

Subpackage QQuantLib
--------------------

.. toctree::
   :maxdepth: 4

   tnbs.BTC_02_AE.QQuantLib
