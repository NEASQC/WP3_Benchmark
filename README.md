# The NEASQC BENCHMARK SUITE (TNBS) Library

This repository is associated to **The NEASQC BENCHMARK SUITE**, **TNBS**, of Work Package 3 of the NEASQC European project. As pointed out in deliverable **D3.5 The NEASQC Benchmark Suite**, each proposed Benchmark Test Case, **BTC**, must have an Atos myqlm compatible software implementation. This Github repository gathers all this code.

## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

There are two ways for installing mandatory Python libraries for using the **TNBS** code: using the *environment.yml* file or the *requirements.txt* file.

* The *environment.yml*: this file contains all the mandatory libraries for executing the **TNBS** code including the different *jupyter-notebooks* presented in the TNBS repository. For installing them the following conda command (new environment called tnbs_test will be created):
    conda env create -n tnbs_test -f environment.yml
* The *requirements.txt*:  this file contains the mandatory libraries for executing the **TNBS** code except the *jupyter-notebooks* (jupyter notebooks libraries should be installed manually). For installing them the following command can be used:
    pip install -r requirements.txt

## Repository organisation

The different **BTC** are located inside the **tnbs** folder:

1. *BTC_01_PL*: Software implementation for the **BTC** of the *Probability Loading kernel*. See **Annex C. T01: Benchmark for Probability Loading Algorithms** from deliverable D3.5.
2. *BTC_02_AE*: Software implementation for the **BTC** of the *Amplitude Estimation kernel*. See **Annex D. T02: Benchmark for Amplitude Estimation Algorithms** from deliverable D3.5.
3. *BTC_03_QPE*: Software implementation for the **BTC** of the *Quantum Amplitude Estimation kernel*. See **Annex E. T03: Benchmark for Phase Estimation Algorithms** from deliverable D3.5.
3. *BTC_04_PH*: Software implementation for the **BTC** of the *Parent Hamiltonian Kernel*. See **Annex F. **T04: Benchmark for Parent Hamiltonian** from deliverable D3.5.

## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation

The html documentation of the **TNBS**  can be access at: https://neasqc.github.io/TNBS
The complete TNBS documentation can be found at the public project deliverable:
[D3.5 The NEASQC Benchmark Suite (TNBS)](https://www.neasqc.eu/wp-content/uploads/2023/10/NEASQC_D3.5_Benchmark_suite_R1.0.pdf)

## Test it

You can test the library in binder using following link:

[Binder Link for TNBS](https://mybinder.org/v2/gh/NEASQC/WP3_Benchmark/HEAD)
