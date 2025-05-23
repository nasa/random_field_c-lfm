import h5py
import os
import numpy as np

def write_sample_data_to_hdf5(hdf5_file_name, wind_samples, x_grid, t_grid, 
                              mean_wind=None, input_params=None, vector_field=True):
    '''
    wind_samples: num_samples x N x 3 x num_time_steps array
    x_grid: N x 3 array
    t_grid: num_time_steps length array
    input_params: dictionary of (param_name, param_values) pairs of different sampling settings (wind direction, frequency grid, etc.)
    '''

    if os.path.exists(hdf5_file_name):
        raise ValueError("File name already exists: " + hdf5_file_name)
    
    with h5py.File(hdf5_file_name, 'w') as fid:
        fid.create_dataset('x_grid', data=x_grid)
        fid.create_dataset('t_grid', data=t_grid)
        if mean_wind is not None:
            fid.create_dataset('mean_wind', data=mean_wind)
        
        samples_group = fid.create_group('wind_samples')
        for i in range(len(wind_samples)):
            sample_i_group = samples_group.create_group(str(i))
            if vector_field:
                for j in range(3):
                    sample_i_group.create_dataset('v{}'.format(j+1), 
                                                  data=wind_samples[i, :, j, :])
            else:
                sample_i_group.create_dataset('v1', data=wind_samples[i, :, :])

        if input_params is not None:
            params_group = fid.create_group('input_parameters')
            for param, value in input_params.items():
                params_group.create_dataset(param, data=np.array(value))


def get_wind_samples_from_hdf5(hdf5_file_name, vector_field=True):
    '''
    Return num_samples x N x 3 x num_time_steps array for full wind velocity field (vector_field=True) or num_samples x N x num_time_steps for scalar wind velocity field
    '''
    with h5py.File(hdf5_file_name, 'r') as fid:
        samples_group = fid['wind_samples']
        N = len(samples_group.keys())
        Nx = len(fid['x_grid'][()])
        Nt = len(fid['t_grid'][()])
        wind_samples = np.zeros((N, Nx, 3, Nt)) if vector_field else np.zeros((N, Nx, Nt))
        for i in range(N):
            sample_i = samples_group[str(i)]
            if vector_field:
                for j in range(3):
                    wind_samples[i, :, j, :] = sample_i['v{}'.format(j+1)][()]
            else:
                wind_samples[i, :, :] = sample_i['v1'][()]
    return wind_samples

def get_x_grid_from_hdf5(hdf5_file_name):
    with h5py.File(hdf5_file_name, 'r') as fid:
        x_grid = fid['x_grid'][()]
    return x_grid

def get_time_grid_from_hdf5(hdf5_file_name):
    with h5py.File(hdf5_file_name, 'r') as fid:
        t_grid = fid['t_grid'][()]
    return t_grid

def get_mean_wind_field_from_hdf5(hdf5_file_name):
    with h5py.File(hdf5_file_name, 'r') as fid:
        mean_wind = fid['mean_wind'][()]
    return mean_wind 

def get_input_params_from_hdf5(hdf5_file_name):
    with h5py.File(hdf5_file_name, 'r') as fid:
        input_group = fid['input_parameters']
        input_params = {}
        for param in input_group.keys():
            input_params[param] = input_group[param][()]
    return input_params
