# Parallel Computing Applications in Finance: Monte Carlo Simulations for Option Pricing

## Overview
This repository demonstrates the application of Monte Carlo simulations in finance for estimating the price of a European call option. The project highlights the use of parallel computing to accelerate simulation processes, comparing the performance and computational efficiency of MATLAB and Python implementations.

### Table of Contents
1. [Project Background and Key Features](#project-background-and-key-features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Results and Conclusion](#results-and-conclusion)
6. [Resources](#resources)
7. [License](#license)
8. [Contact Information](#contact-information)

## Project Background and Key Features
Monte Carlo simulation is a widely used computational algorithm in finance for modeling asset price behavior and estimating the value of financial derivatives, such as options. This project employs Monte Carlo simulation to estimate the price of a European call option by simulating multiple paths for the underlying asset price. The results are then compared with the theoretical price derived from the Black-Scholes formula.

### Key Features
- **Black-Scholes Formula:** The theoretical price of a European call option is calculated using the Black-Scholes formula, and the results are compared with the Monte Carlo estimated price to evaluate accuracy.
- **Monte Carlo Simulation:** A numerical method used for option pricing, which approximates the value of an option by simulating multiple possible future paths of the underlying asset's price.
- **Parallel Computing:** Implementations in both MATLAB and Python leverage parallel computing to improve the computational efficiency of the Monte Carlo simulations.
- **Comparative Analysis:** The repository includes a comparative analysis of the runtime and accuracy between MATLAB and Python implementations, both in parallel and non-parallel configurations.


## Project Structure 
The project is organized into the following files:
- `README.md`: Provides an overview of the project, usage instructions, and other relevant details.
- `OptionPricing_FinalReview.mlx`: MATLAB live script for running the Monte Carlo simulations and comparing the results with the Black-Scholes formula.
- `monte_carlo_multiprocessing.py`: Python script for running the Monte Carlo simulations using multiprocessing.
- `MATLAB_Results.mat`: Pre-computed MATLAB results from the Monte Carlo simulations.
- `Python_Results.mat`: Pre-computed Python results from the Monte Carlo simulations.
- `runtime_comparison.png`: An image file showing the runtime comparison between MATLAB and Python implementations.

## Requirements
- **MATLAB**: R2024a or later with Parallel Computing Toolbox.
- **Python**: Version 3.9 to 3.11 with NumPy, SciPy, Joblib, Pandas, Multiprocessing
, and Pathos.
- **Compatibility**: Ensure Python version is compatible with MATLAB release. See [MATLAB Python Compatibility Guide](https://www.mathworks.com/support/requirements/python-compatibility.html).


## Usage

### 1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/monte-carlo-option-pricing.git
   cd monte-carlo-option-pricing
   ```

### 2. MATLAB
#### Option 1: Use Pre-Computed Results
If you do not wish to re-run the simulations and prefer to use pre-computed results:
1. Open the `OptionPricing_FinalReview.mlx` script in MATLAB.
2. Set the `doComputing` flags to `false` within the script to load and display the pre-computed results stored in `MATLAB_Results.mat` and `Python_Results.mat`.
3. Run the script to view the results and runtime comparison plots.
#### Option 2: Re-run the Simulations
If you wish to re-run the Monte Carlo simulations:
1. Open the `OptionPricing_FinalReview.mlx` script in MATLAB.
2. Set the `doComputing` flags to `true` within the script to run the simulations from scratch.
3. Run the script to perform the Monte Carlo simulations and compare the results against the theoretical prices calculated using the Black-Scholes formula.

### 3. Python
#### Option 1: Use Pre-Computed Results
If you do not wish to re-run the simulations and prefer to use pre-computed results:
1. Ensure you have Python version 3.9 to 3.11 installed on your machine.
2. Open the `monte_carlo_multiprocessing.py` script in your preferred IDE or text editor.
3. Set the `doComputingPython` flag to `False` to load and display the pre-computed results stored in `Python_Results.mat`.
#### Option 2: Re-run the Simulations
If you wish to re-run the Monte Carlo simulations:
1. Ensure you have Python version 3.9 to 3.11 installed on your machine.
2. Install the required Python packages 
3. Open the `monte_carlo_multiprocessing.py` script in your preferred IDE or text editor. 
4. Set the `doComputingPython` flag to `True` to perform the simulations from scratch. 
5. Run the script to perform the Monte Carlo simulations.


## Results and Conclusion
This project demonstrates the effectiveness of Monte Carlo simulations for option pricing and the significant performance gains achievable through parallel computing. MATLAB's optimized numerical libraries and parallel processing capabilities make it a preferred choice for complex financial modeling tasks, though Python remains a flexible alternative.
### Runtime Comparison
Below is a runtime comparison between MATLAB and Python implementations in both parallel and non-parallel configurations:
![Runtime Comparison](runtime_comparison.png)


## Resources
- [MATLAB Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)
- [What Is Monte Carlo Simulation?](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information
For any questions, suggestions, or collaboration inquiries, feel free to reach out:
- [jl6649@columbia.edu](mailto:jl6649@columbia.edu)
- [lc3813@columbia.edu](mailto:lc3813@columbia.edu)