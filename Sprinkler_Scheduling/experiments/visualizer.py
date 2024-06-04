from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import interpolate

import matplotlib.animation
import matplotlib.cm as cm
import matplotlib.colors as colors

# Define animate function for both subplots
def visual(logger):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]

    ax1.set_title('env and trajectory')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    ax2.set_title('MI')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    ax3.set_title('Observed env')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')

    ax4.set_title('Computed effect')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')

    # Set axis limits for both subplots
    ax1.axis([-0.5, 19.5, -0.5, 19.5])
    ax2.axis([-0.5, 19.5, -0.5, 19.5])
    ax3.axis([-0.5, 19.5, -0.5, 19.5])
    ax4.axis([-0.5, 19.5, -0.5, 19.5])

    # Invert y-axis for both subplots
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()

    # Create heatmap objects for both subplots
    heatmap1 = logger.save_data['truth_env'][0]
    heatmap2 = logger.save_data['MI_information'][0]
    heatmap3 = logger.save_data['observed_env'][0]
    heatmap4 = logger.save_data['computed_effect'][0]

    im1 = ax1.imshow(heatmap1, cmap=cm.coolwarm, interpolation='nearest', origin='lower')
    im2 = ax2.imshow(heatmap2, cmap=cm.coolwarm, interpolation='nearest', origin='lower')
    im3 = ax3.imshow(heatmap3, cmap=cm.coolwarm, interpolation='nearest', origin='lower')
    im4 = ax4.imshow(heatmap4, cmap=cm.coolwarm, interpolation='nearest', origin='lower')

    # Add colorbars to both subplots
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar4 = fig.colorbar(im4, ax=ax4)

    # Plot the initial trajectory on the third subplot
    l_list = []
    arr = dict()
    for id in range(logger.save_data['info']['team_size']):
        arr[id+1] = []
    for step in range(len(logger.save_data['time_series'])):
        for id in range(len(logger.save_data['time_series'][step]['state'])):
            state = logger.save_data['time_series'][step]['state'][id+1]
            arr[id+1].append(state)
    for id in range(logger.save_data['info']['team_size']):     
        arr[id+1] = np.array(arr[id+1])
        # l, = ax1.plot([],[], color='black', linewidth=1)
        l, = ax1.plot([],[], color='black', linewidth=1 + 0.5*id)
        l_list.append(l)

    # Define animate function for both subplots
    def animate(i):
        heatmap1 = logger.save_data['truth_env'][i]
        heatmap2 = logger.save_data['MI_information'][i]
        heatmap3 = logger.save_data['observed_env'][i]
        heatmap4 = logger.save_data['computed_effect'][i]
        
        im1.set_data(heatmap1)
        im2.set_data(heatmap2)
        im3.set_data(heatmap3)
        im4.set_data(heatmap4)
        
        # Adjust the color range of the heatmap
        im1.set_clim(vmin=20, vmax=200)
        im2.set_clim(vmin=0, vmax=1.0)
        im3.set_clim(vmin=0, vmax=180)
        im4.set_clim(vmin=0, vmax=50)
        
        for id in range(logger.save_data['info']['team_size']):
            traj = arr[id+1]
            # Extract the x-coordinates and y-coordinates up to time i
            x = traj[:i+1, 0]
            y = traj[:i+1, 1]
            # Update the trajectory with the current x and y coordinates
            l_list[id].set_data(y, x)

        return im1, im2, im3, im4, l_list

    # Create animation object for both subplots
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames = len(logger.save_data['time_series']))

    return ani