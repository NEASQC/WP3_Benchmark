BTC\_04\_PH: Parent Hamiltonian
===============================

This part of the repository is associated with the Benchmark Test Case (BTC) 04: **Parent Hamiltonian**. 
Configure the **kernel_configuration** dictionary in final part of the **my_benchmark_execution.py** file. The number of qubits for executing should be provided as a python list to the variable: *list_of_qbits*. The different depths for executing should be provided as a python list to the key *depth* of the **kernel_configuration** dictionary.

* The mandatory configuration files for the ansatz (they have the word *parameters* in their names) and for the different Pauli decomposition (they have the word *Pauli* in their names) can be found under the **configuration_files** folder.
* The number of qubits for the ansatz could be from 3 to 30 qubits.
* The depth of the ansatz can be from 1 to 4.

The files will be saved in the **Results** folder (you can change the folder in key argument **saving_folder** of the **benchmark_arguments** dictionary). Then you should execute the following command line:

    python my_benchmark_execution.py

If you want a complete execution and create the complete **JSON** file with the complete information of the benchmark results following the **NEASQC JSON Schema** you can instead execute the script **benchmark_exe.sh**:
 
    bash benchmark_exe.sh

All the results files and the corresponding JSON will be stored in the **Results** folder.

my\_benchmark\_execution
------------------------

.. automodule:: tnbs.BTC_04_PH.my_benchmark_execution
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_info
-------------------

.. automodule:: tnbs.BTC_04_PH.my_benchmark_info
   :members:
   :undoc-members:
   :show-inheritance:

my\_benchmark\_summary
----------------------

.. automodule:: tnbs.BTC_04_PH.my_benchmark_summary
   :members:
   :undoc-members:
   :show-inheritance:

my\_environment\_info
---------------------

.. automodule:: tnbs.BTC_04_PH.my_environment_info
   :members:
   :undoc-members:
   :show-inheritance:

neasqc\_benchmark
-----------------

.. automodule:: tnbs.BTC_04_PH.neasqc_benchmark
   :members:
   :undoc-members:
   :show-inheritance:


Subpackage PH
-------------

.. toctree::
   :maxdepth: 4
   :hidden:

   tnbs.BTC_04_PH.PH
