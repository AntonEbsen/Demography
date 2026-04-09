import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import statsmodels.api as sm
import numpy as np

def plot_map(gdf, column, title, cmap='viridis', legend=True):
    """
    Standardized geospatial visualization for the project.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(column=column, 
             cmap=cmap, 
             legend=legend, 
             ax=ax,
             legend_kwds={'label': column, 'orientation': "horizontal"})
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_parallel_trends(df, year_col, outcome_col, treatment_col):
    """
    Visualizes trends for treatment and control groups to assess parallel trends assumption.
    """
    plt.figure(figsize=(10, 6))
    trends = df.groupby([year_col, treatment_col])[outcome_col].mean().reset_index()
    sns.lineplot(data=trends, x=year_col, y=outcome_col, hue=treatment_col, marker='o', linewidth=2.5)
    plt.title(f'Parallel Trends: Mean {outcome_col} ({df[year_col].min()}-{df[year_col].max()})', fontsize=14)
    plt.ylabel(f'Mean {outcome_col}')
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_diagnostics(results, df_reg, corr_vars):
    """
    Faceted plot including Correlation Heatmap, Residuals vs Fitted, Normal Q-Q, and Cook's Distance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Correlation Matrix Heatmap
    sns.heatmap(df_reg[corr_vars].corr(), annot=True, cmap='coolwarm', ax=axes[0,0])
    axes[0,0].set_title('Correlation Matrix of Regression Variables')

    # 2. Residuals vs Fitted Plot
    sns.residplot(x=results.fittedvalues, y=results.resid, lowess=True, ax=axes[0,1], 
                  scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0,1].set_title('Residuals vs Fitted Values')
    axes[0,1].set_xlabel('Fitted Values')
    axes[0,1].set_ylabel('Residuals')

    # 3. Normal Q-Q Plot
    sm.qqplot(results.resid, line='45', ax=axes[1,0])
    axes[1,0].set_title('Normal Q-Q Plot')

    # 4. Cook's Distance
    influence = results.get_influence()
    (c, p) = influence.cooks_distance
    axes[1,1].stem(np.arange(len(c)), c, markerfmt=",")
    axes[1,1].set_title("Cook's Distance per Observation")
    axes[1,1].set_xlabel('Observation Index')
    axes[1,1].set_ylabel('Cook\'s Distance')

    plt.tight_layout()
    return fig
