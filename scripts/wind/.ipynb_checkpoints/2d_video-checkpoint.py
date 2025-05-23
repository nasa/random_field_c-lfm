import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    f = h5py.File('data/wind/wind_samples_scalar_high-res_N5000.hdf5', 'r')
    t_grid = np.array(f['t_grid'])
    x_grid = np.array(f['x_grid'])

    y = x_grid[:, 1]
    z = x_grid[:, 2]

    print(y.shape)
    print(z.shape)

    u = np.array(f['wind_samples']['50']['v1']).reshape(10, 10, 256)
    print(u[:, 1:, :].min(), u[:, 1:, :].max())

    vmin = 20
    vmax = 45
    colormap = 'cool'

    fig, ax = plt.subplots(1, 1)
    Q = fig.colorbar(ax.pcolor(u[:, 1:, 0].T, vmin = vmin, vmax = vmax, cmap = colormap))

    def update(i):
        global Q
        Q.remove()
        Q = fig.colorbar(ax.pcolor(u[:, 1:, i].T, vmin = vmin, vmax = vmax, cmap = colormap))
        print(i)
        
    ani = FuncAnimation(fig, update, frames = range(256))
    ani.save('wind.gif', fps = 30)