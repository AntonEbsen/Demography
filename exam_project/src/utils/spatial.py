import libpysal
from esda.moran import Moran, Moran_Local
import splot.esda as esda_plot
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def calculate_spatial_weights(gdf):
    """
    Generates a row-standardized Queen contiguity weights matrix.
    """
    logger.info("Generating Queen contiguity spatial weights...")
    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = 'r'
    return w

def run_global_moran(gdf, variable, w):
    """
    Calculates Global Moran's I for a given variable.
    """
    logger.info(f"Calculating Global Moran's I for {variable}...")
    moran = Moran(gdf[variable], w)
    return {
        'I': moran.I,
        'p-value': moran.p_sim,
        'z-score': moran.z_sim
    }

def plot_lisa_clusters(gdf, variable, w, output_path=None):
    """
    Visualizes Local Indicators of Spatial Association (LISA).
    """
    logger.info(f"Generating LISA cluster map for {variable}...")
    moran_loc = Moran_Local(gdf[variable], w)
    fig, ax = esda_plot.lisa_cluster(moran_loc, gdf, p=0.05, figsize=(10, 8))
    plt.title(f"LISA Clusters: {variable}")
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"LISA plot saved to {output_path}")
    
    return fig
