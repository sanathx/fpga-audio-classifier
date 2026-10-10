\# FPGA-Accelerated Audio Classification



A hardware–software project exploring fixed-point FFT implementation in SystemVerilog and audio analysis using Python-based signal processing and machine learning.



\## Overview



The project implements a 256-point Fast Fourier Transform (FFT) in SystemVerilog and includes Python tools for generating test signals, analysing audio, and comparing hardware FFT outputs against a NumPy reference.



The aim is to explore how spectral processing can be implemented in RTL and integrated into a broader audio classification workflow.



\## Architecture



The project has two main components:



\- \*\*RTL:\*\* A 256-point radix-2 FFT implemented using SystemVerilog modules for butterfly operations, complex multiplication, twiddle factors, and FFT stages.

\- \*\*Python:\*\* Scripts for signal generation, fixed-point FFT modelling, reference calculations, audio processing, model training, and evaluation.



\## Hardware Implementation



The FFT design uses:



\- 256-point radix-2 decimation-in-time (DIT) FFT

\- Signed fixed-point arithmetic with Q-format twiddle factors

\- Eight FFT stages

\- Dedicated butterfly and complex multiplier modules

\- A twiddle-factor ROM containing precomputed coefficients



The current RTL connects the eight stages combinationally. It is \*\*not a clocked, register-pipelined implementation\*\*.



\## Simulation and Verification



The Vivado testbench reads complex input samples from a text file, runs them through the FFT design, and writes the output samples to a second file.



Python scripts provide reference FFT calculations and numerical comparisons against NumPy.



Numerical accuracy and end-to-end classification performance should be evaluated using the supplied scripts and test data before claiming specific performance figures.



\## Software



\- Python

\- NumPy

\- SystemVerilog

\- AMD Xilinx Vivado

\- scikit-learn and other Python libraries used by the model scripts



\## Repository Structure



```text

.

├── README.md

├── .gitignore

├── python/

│   ├── fixed\_point\_fft.py

│   ├── fft\_reference.py

│   ├── compare\_fft.py

│   └── ...

└── project\_1/

&#x20;   ├── project\_1.xpr

&#x20;   └── project\_1.srcs/

&#x20;       ├── sources\_1/new/

&#x20;       └── sim\_1/new/

```



\## Getting Started



1\. Clone the repository.

2\. Open `project\_1/project\_1.xpr` in Vivado.

3\. Review the simulation sources and configure the testbench file paths for your local environment.

4\. Run the FFT simulation with the required input samples.

5\. Use the Python reference and comparison scripts to analyse the outputs.



Python dependencies vary by script. Install the libraries required by the particular workflow you intend to run.



\## Current Limitations



\- The RTL FFT is combinational rather than clocked and pipelined.

\- The testbench currently uses machine-specific Windows file paths that need to be made portable.

\- Dataset files and locally generated outputs are excluded from version control.

\- Accuracy, resource utilisation, timing, and classification performance should be measured before reporting quantitative results.



\## Future Work



\- Make simulation file paths portable.

\- Validate fixed-point RTL results against the Python reference.

\- Measure synthesis resource utilisation and timing.

\- Integrate and evaluate the complete audio classification pipeline.

.

