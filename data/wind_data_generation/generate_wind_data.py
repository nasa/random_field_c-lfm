import numpy as np
from pathlib import Path

from wind_sampler import BasicWindSampler
from hdf5_sample_data import write_sample_data_to_hdf5


if __name__ == "__main__":
    """
    Script to generate train and test data for the wind field reconstruction problem with C-LFM. Run with seed=0 and create '../wind/wind_train_data.hdf5' then seed=1 and create '../wind/wind_test_data.hdf5' to be able to run train & test scripts in the scripts/wind directory.

    Note: scripts currently assume that train and test data are created on 10x10 spatial grids, so modifications will be needed if Ny and Nz are changed below.
    """

    np.random.seed(0)
    output_file = Path(__file__).parent / "../wind/wind_train_data.hdf5"
    num_samples = 5000
    # np.random.seed(1)
    # output_file = Path(__file__).parent / "../wind/wind_test_data.hdf5"
    # num_samples = 1000

    Ny = 10  # num grid points in x2 / x3 directions
    Nz = 10
    H = 100
    W = 100
    ZMIN = 1

    x2_grid = np.linspace(0, W, Ny)
    x3_grid = np.linspace(ZMIN, H, Nz)
    x_coords = np.zeros((Ny * Nz, 3))
    x_coords[:, 1] = np.repeat(x2_grid, Nz)
    x_coords[:, 2] = np.tile(x3_grid, Ny)

    # Define frequency and time grid parameters:
    F = 3.0
    N_w = 2**7
    delta_w = F / N_w
    # T needs to be defined this way:
    T = 1 / delta_w
    N_t = 2 * N_w
    delta_t = T / N_t
    time_steps = np.linspace(0, T - delta_t, N_t)
    freqs = np.linspace(delta_w, F, N_w)

    # parameters from the paper:
    alphas = np.array([1.0, 0.75, 0.25])
    betas = np.array([1.0, 0.25, 0.1])
    shear_velocity = 1.9

    wind_sampler = BasicWindSampler(x_coords, freqs, shear_velocity)
    samples = wind_sampler.sample(num_samples, delta_t, N_t)

    params = {"freq_grid": freqs, "shear_velocity": shear_velocity}
    write_sample_data_to_hdf5(
        output_file,
        samples,
        x_coords,
        time_steps,
        input_params=params,
        vector_field=False,
    )
