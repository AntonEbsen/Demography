import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Cost of Quality Dashboard",
    page_icon="👶",
    layout="wide",
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4253;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    processed_path = Path('exam_project/data/processed/master_panel_data.csv')
    if not processed_path.exists():
        st.error("Processed data not found. Please run 'python exam_project/src/data/process_data.py' first.")
        return None
    return pd.read_csv(processed_path)

@st.cache_data
def load_geo(year):
    geo_path = Path(f'exam_project/data/raw/data{year}_0.geojson')
    if not geo_path.exists():
        return None
    return gpd.read_file(geo_path)

df_panel = load_data()

# --- Sidebar ---
st.sidebar.title("🔍 Search Filters")
if df_panel is not None:
    years = sorted(df_panel['Year'].unique())
    selected_year = st.sidebar.selectbox("Select View Year", years, index=0)
    
    analysis_var = st.sidebar.selectbox(
        "Metric to Visualize",
        ["TFR", "F_TEX", "F_CL_1013", "IMR"],
        help="TFR: Total Fertility Rate, F_TEX: Textile Share, F_CL_1013: Child Labor"
    )

# --- Header ---
st.title("🛡️ The 'Cost of Quality'")
st.subheader("Legislative Shocks and the British Fertility Transition (1851-1881)")

if df_panel is not None:
    # --- Metrics row ---
    year_data = df_panel[df_panel['Year'] == selected_year]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg TFR", round(year_data['TFR'].mean(), 2))
    with col2:
        st.metric("Textile Intensity", f"{round(year_data['F_TEX'].mean(), 2)}%")
    with col3:
        st.metric("Child Labor Share", f"{round(year_data['F_CL_1013'].mean(), 2)}%")
    with col4:
        st.metric("Districts reporting", len(year_data))

    # --- Main Visuals ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Geospatial Distribution", 
        "📈 Temporal Trends", 
        "📖 District Deep Dive",
        "🔬 Policy Simulator",
        "🤖 AI Research Insights",
        "🛠️ Research Model Builder",
        "📋 Export Results"
    ])
    
    with tab1:
        st.write(f"### Historical Map: {analysis_var} in {selected_year}")
        geo_df = load_geo(selected_year)
        if geo_df is not None:
            # Merge with panel data
            merged = geo_df.merge(year_data, on='REGDIST', how='left')
            
            # Interactive Map with Folium
            import folium
            from streamlit_folium import st_folium
            
            m = folium.Map(location=[52.5, -1.5], zoom_start=6, tiles="CartoDB dark_matter")
            
            folium.Choropleth(
                geo_data=merged,
                name="choropleth",
                data=merged,
                columns=["REGDIST", analysis_var],
                key_on="feature.properties.REGDIST",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f"{analysis_var} Value",
            ).add_to(m)
            
            st_folium(m, width=900, height=600, key=f"map_{selected_year}_{analysis_var}")
        else:
            st.info("GeoJSON for this year not available in data/raw.")

    with tab2:
        st.write("### Comparative Trends by Textile Intensity")
        base_year = df_panel[df_panel['Year'] == 1851]
        median_tex = base_year['F_TEX'].median()
        high_tex_dists = base_year[base_year['F_TEX'] > median_tex]['REGDIST'].unique()
        
        df_panel['Type'] = df_panel['REGDIST'].apply(lambda x: 'Textile-Heavy' if x in high_tex_dists else 'Agricultural')
        
        trend_df = df_panel.groupby(['Year', 'Type'])[analysis_var].mean().reset_index()
        fig_trend = px.line(trend_df, x='Year', y=analysis_var, color='Type', markers=True,
                           template="plotly_dark", title=f"Trend of {analysis_var} Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.write("### 📖 District Research Biography")
        all_districts = sorted(df_panel['REGDIST'].unique())
        selected_dist = st.selectbox("Search Registration District", all_districts, index=0)
        
        dist_data = df_panel[df_panel['REGDIST'] == selected_dist]
        
        col_dist1, col_dist2 = st.columns([2, 1])
        with col_dist1:
            st.write(f"#### Trend Comparison: {selected_dist}")
            # Get national average for the same variable
            national_avg = df_panel.groupby('Year')[analysis_var].mean().reset_index()
            national_avg['REGDIST'] = 'National Average'
            
            comparison_df = pd.concat([
                dist_data[['Year', 'REGDIST', analysis_var]],
                national_avg
            ])
            
            fig_dist = px.line(comparison_df, x='Year', y=analysis_var, color='REGDIST', markers=True,
                               template="plotly_dark", title=f"{analysis_var}: {selected_dist} vs. National Mean")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_dist2:
            st.write("#### District Snapshot")
            
            # Boundary Stability Warning
            stability_path = Path('exam_project/outputs/diagnostics/boundary_stability_report.csv')
            if stability_path.exists():
                stability_df = pd.read_csv(stability_path)
                selected_stab = stability_df[(stability_df['REGDIST'] == selected_dist) & (stability_df['Year'] == selected_year)]
                if not selected_stab.empty and selected_stab['Unstable'].iloc[0]:
                    st.warning("⚠️ **Boundary Instability Detected**: This district underwent significantly administrative changes since 1851 (>5% area change). Interpret results with caution.")

            latest_dist = dist_data[dist_data['Year'] == selected_year]
            if not latest_dist.empty:
                st.metric("District TFR", latest_dist['TFR'].iloc[0])
                st.metric("Textile Share", f"{latest_dist['F_TEX'].iloc[0]}%")
                st.metric("Child Labor (F)", f"{latest_dist['F_CL_1013'].iloc[0]}%")
            else:
                st.warning(f"No data for {selected_dist} in {selected_year}")

    with tab4:
        st.write("### 🔬 Legislative Impact Simulator")
        st.markdown("""
        Estimate how changes in child labor enforcement and health outcomes might have shifted the 
        fertility transition based on our econometric model.
        """)
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            reduction_cl = st.slider(
                "Enforcement Strength (Child Labor Reduction %)", 
                0, 100, 20, 
                help="How much the Factory Acts reduce child labor participation."
            )
        with col_sim2:
            improvement_imr = st.slider(
                "Public Health Progress (IMR Reduction %)", 
                0, 50, 10,
                help="Reduction in infant mortality through sanitation or medical access."
            )
            
        # Model Coefficients (from exam_project.ipynb)
        b_cl = 0.0036
        b_imr = 0.0023
        
        # Calculate shifts
        cl_delta = (avg_cl := year_data['F_CL_1013'].mean()) * (reduction_cl / 100)
        imr_delta = (avg_imr := year_data['IMR'].mean()) * (improvement_imr / 100)
        
        tfr_delta = (cl_delta * b_cl) + (imr_delta * b_imr)
        
        st.metric(
            "Predicted Change in TFR", 
            f"-{round(tfr_delta, 3)}", 
            delta_color="normal"
        )
        
        st.info(f"**Insight:** A {reduction_cl}% reduction in child labor alone would account for {round(cl_delta * b_cl / tfr_delta * 100 if tfr_delta > 0 else 0, 1)}% of the predicted fertility decline.")

    with tab5:
        st.write("### 🤖 Interpretable AI: SHAP Global Insights")
        st.markdown("""
        While our econometric models focus on causal identification, these **Machine Learning Insights** 
        use a Random Forest model to rank the global predictive power of each feature.
        """)
        
        shap_plot_path = Path('exam_project/outputs/ml/shap_summary_plot.png')
        if shap_plot_path.exists():
            st.image(str(shap_plot_path), caption="SHAP Summary Plot: Higher magnitude indicates stronger predictive influence.")
        else:
            st.warning("SHAP analysis results not found. Please run 'python exam_project/src/analysis/interpret.py' first.")
            
        st.write("#### 📑 How to read this plot")
        st.info("""
        - **Feature Position**: Higher features on the Y-axis are more important overall.
        - **Color**: Red indicates high values of the feature, Blue indicates low values.
        - **Impact (X-axis)**: The further a dot is from zero, the larger its impact on predicting TFR.
        """)
    
    with tab6:
        st.write("### 🛠️ Interactive Research Model Builder")
        st.markdown("Build and estimate your own OLS model on the master panel dataset.")
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            dep_var = st.selectbox("Dependent Variable (Y)", ["TFR", "F_CL_1013", "IMR"])
            indep_vars = st.multiselect(
                "Predictors (X)", 
                ["F_TEX", "IMR", "F_CL_1013", "Year", "dist_to_hub"],
                default=["F_TEX", "Year"]
            )
            use_fe = st.checkbox("Include Year Fixed Effects", value=True)
            cluster_level = st.selectbox("Standard Error Clustering", ["None", "REGCNTY", "REGDIST"])
            
        with col_m2:
            if indep_vars:
                import statsmodels.formula.api as smf
                # Construct Formula
                formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
                if use_fe and "Year" not in indep_vars:
                    formula += " + C(Year)"
                
                # Run Model
                try:
                    if cluster_level != "None" and cluster_level in df_panel.columns:
                        res = smf.ols(formula, data=df_panel).fit(
                            cov_type='cluster', 
                            cov_kwds={'groups': df_panel[cluster_level]}
                        )
                    else:
                        res = smf.ols(formula, data=df_panel).fit()
                        
                    st.write("#### OLS Results")
                    st.code(res.summary().as_latex(), language="latex")
                    st.info("Copy the LaTeX above directly into your paper.")
                except Exception as e:
                    st.error(f"Model Error: {e}")
            else:
                st.info("Select at least one predictor to see results.")
        
    with tab7:
        st.write("### 📥 Download Research Artifacts")
        csv = year_data.describe().to_csv().encode('utf-8')
        st.download_button(
            label=f"Download {selected_year} Summary Stats (CSV)",
            data=csv,
            file_name=f'summary_stats_{selected_year}.csv',
            mime='text/csv',
        )
        
        st.write("---")
        st.write("#### LaTeX Export")
        if st.button("Generate LaTeX Summary Table"):
            st.code(year_data.describe().to_latex(), language="latex")

    # --- Insight Section ---
    st.divider()
    st.write("### 📜 Methodology Note")
    st.info("""
    This dashboard represents the exploratory phase of the 'Cost of Quality' research. 
    The interactive maps allow for granular exploration of registration districts across 
    the transformation of the British labor market.
    """)
else:
    st.warning("Please ensure the data processing script has been executed to populate the dashboard.")
