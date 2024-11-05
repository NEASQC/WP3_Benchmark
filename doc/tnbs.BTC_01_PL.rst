BTC\_01\_PL: Probability Loading
================================

This part of the repository is associated with the Benchmark Test Case (BTC) 01: **Probability Loading**. 

Configure the **kernel_configuration** dictionary at the end of the **my_benchmark_execution.py** file. The files will be saved in the **Results** folder (you can change the folder in key argument **saving_folder** of the **benchmark_arguments** dictionary). Then you should execute the following command line:

    python my_benchmark_execution.py

If you want a complete execution and create the complete **JSON** file with the complete information of the benchmark results following the **NEASQC JSON Schema** you can instead execute the script **benchmark_exe.sh**:
 
    bash benchmark_exe.sh

All the results files and the corresponding JSON will be stored in the **Results** folder.

my\_benchmark\_execution
------------------------

.. automodule:: tnbs.BTC_01_PL.my_benchmark_execution
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_info
-------------------

.. automodule:: tnbs.BTC_01_PL.my_benchmark_info
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_summary
----------------------

.. automodule:: tnbs.BTC_01_PL.my_benchmark_summary
   :members:
   :undoc-members:
   :show-inheritance:

my\_environment\_info
---------------------

.. automodule:: tnbs.BTC_01_PL.my_environment_info
   :members:
   :undoc-members:
   :show-inheritance:

neasqc\_benchmark
-----------------

.. automodule:: tnbs.BTC_01_PL.neasqc_benchmark
   :members:
   :undoc-members:
   :show-inheritance:

Subpackage PL
-------------

.. toctree::
   :maxdepth: 4
   :hidden:

   tnbs.BTC_01_PL.PL
