# Inference of Diffusive Model Dynamics in Sparse Data Regimes through Simulation-Based Methods

This repository contains code and notebooks for simulating and analyzing diffusive model dynamics using simulation-based inference (SBI) methods.

### Notebooks

- **Brownian_Motion_SBI_embedding.ipynb**: Simulates Brownian motion and uses SBI for embedding.
- **Brownian_Motion_SBI_transition_matrix.ipynb**: Simulates Brownian motion and uses SBI with a transition matrix.
- **joint_posterior_advanced_metrics.ipynb**: Analyzes joint posterior distributions with advanced metrics.
- **langevin_integrator_SBI.ipynb**: Simulates Langevin dynamics and uses SBI.
- **plotting_code.ipynb**: Contains code for plotting results.
- **posterior_missspecification.ipynb**: Analyzes posterior misspecification.
- **potential_plot.ipynb**: Plots potential functions.
- **sbc.ipynb**: Performs simulation-based calibration (SBC).

### Source Code

- **src/mamba.py**: Contains the implementation of the Mamba model.
- **src/pscan.py**: Contains the implementation of the pscan algorithm.
- **src/temporal_encoders.py**: Contains temporal encoding functions.

## Getting Started

1. Clone the repository:
    ```sh
    git clone [https://github.com/yourusername/your-repo.git](https://github.com/atemynany/Inference-of-Diffusive-Model-Dynamics-in-Sparse-Data-Regimes-through-Simulation-Based-Methods/new/main?filename=README.md)
    cd your-repo
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the notebooks using Jupyter:
    ```sh
    jupyter notebook
    ```
