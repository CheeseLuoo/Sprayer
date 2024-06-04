from PIL import Image
import numpy as np
from pathlib import Path
from urllib import request
from skimage import transform
from matplotlib import pyplot as plt
# from .visualizer import create_colorbar_ax, OOMFormatter, order_of_magnitude
import pde
import pandas as pd

class DataLoader:
    """Load data as numpy array."""

    def __init__(self, data_path: str) -> None:
        """

        Parameters
        ----------
        data_path: str
            Path to the data file.

        """
        self.data_path = data_path
        self.read_data()
        print(f"Successfully read data from {self.data_path}")
        print(f"Array \tshape:\t{self.array.shape}")
        print(f"\tmin:\t{self.array.min(): .2f}")
        print(f"\tmax:\t{self.array.max(): .2f}")
        print(f"\tdtype:\t{self.array.dtype}\n")

    def read_data(self) -> None:
        """Read data from various formats."""
        file_extension = self.data_path.split(".")[-1]
        if file_extension == "png":
            self.read_png()
        elif file_extension == "npy":
            self.read_npy()

    def read_png(self) -> None:
        """Read data from png image."""
        image = Image.open(self.data_path).convert("L")
        self.array = np.array(image).astype(np.float64)

    def read_npy(self) -> None:
        """Read data from npy file."""
        self.array = np.load(self.data_path).astype(np.float64)

    def get_data(self) -> np.ndarray:
        """Return data as numpy array."""
        return self.array


def download_environment(name, data_path):
    path = Path(f"{data_path}/raw/{name}.jpg")
    if not path.is_file():
        print(f"Downloading to {path}...this step might take some time.")
        request.urlretrieve(
            url="https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
            + f"{name}.SRTMGL1.2.jpg",
            filename=path,
        )
        print("Done")


def preprocess_environment(name, data_path):
    print(f"Preprocessing {name}.jpg...")
    image = Image.open(f"{data_path}/raw/{name}.jpg").convert("L")
    array = np.array(image).astype(np.float64)
    resized = transform.resize(array, (
        array.shape[0] // 10,
        array.shape[1] // 10,
    ))
    save_path = f"{data_path}/preprocessed/{name}.npy"
    np.save(save_path, resized)
    print(f"Saved to {save_path}.")


def get_environment(name, data_path):
    path = Path(f"{data_path}/preprocessed/{name}.npy")
    if not path.is_file():
        Path(f"{data_path}/raw").mkdir(parents=True, exist_ok=True)
        download_environment(name, data_path)
        Path(f"{data_path}/preprocessed").mkdir(parents=True, exist_ok=True)
        preprocess_environment(name, data_path)
    data_loader = DataLoader(str(path))
    return data_loader.get_data()



class Diffusion_Model():
    def __init__(self, Setting):
        self.diffusivity = Setting.diffusivity_K
        self.x_range = Setting.grid_x
        self.y_range = Setting.grid_y
        self.grid = pde.UnitGrid([Setting.grid_x, Setting.grid_y])
        self.R_field = pde.ScalarField(grid = self.grid, data = Setting.R)
        self.initial_field = pde.ScalarField(grid = self.grid, data = Setting.env)
        self.PHY_PDE = DiffusionPDE_withR(diffusivity=self.diffusivity, R = self.R_field)
        
    def solve(self, t):
        return self.PHY_PDE.solve(self.initial_field, t_range = t).data

class DiffusionPDE_withR(pde.PDEBase):
    def __init__(self, diffusivity=1, bc="auto_periodic_neumann", bc_laplace="auto_periodic_neumann", R = 0):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.diffusivity = diffusivity
        self.bc = bc
        self.bc_laplace = bc_laplace
        self.R = R
        
    def evolution_rate(self, state, t=0):
        """ numpy implementation of the evolution equation """
        state_lapacian = state.laplace(bc=self.bc)
        state_gradient = state.gradient(bc=self.bc)
        return (self.diffusivity * state_lapacian
                + self.R - 0.1*state)
