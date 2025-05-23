import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import ot

def show_corner(x: Array, cmap: str = 'Blues', binrange = (-3,3), **kwargs) -> plt.Figure:
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in range(16, cmap.N)]
    colors = [(1.0, 1.0, 1.0), *colors]
    cmap = plt.cm.colors.ListedColormap(colors)

    return sb.pairplot(
        data=pd.DataFrame({f'$x_{i}$': xi for i, xi in enumerate(np.asarray(x).T)}),
        corner=True,
        kind='hist',
        plot_kws={'bins': 64, 'binrange': binrange, 'thresh': None, 'cmap': cmap},
        diag_kws={'bins': 64, 'binrange': binrange, 'element': 'step', 'color': cmap(cmap.N // 2)},
        **kwargs,
    )
