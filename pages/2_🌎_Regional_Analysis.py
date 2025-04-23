import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import traceback
from typing import Optional, Dict, List, Union

st.set_page_config(page_title="Regional Analysis", page_icon="🌎", layout="wide")

# Custom CSS for better error messages and UI consistency with other pages
st.markdown("""
    <style>
    .error-message {
        color: #e74c3c;
        padding: 1rem;
        background-color: #fadbd8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-message {
        color: #2980b9;
        padding: 1rem;
        background-color: #d6eaf8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-message {
        color: #27ae60;
        padding: 1rem;
        background-color: #d4efdf;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Display error function for consistent error handling
def display_error(message: str) -> None:
    """Display formatted error message"""
    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

# Display info function for consistent info messaging
def display_info(message: str) -> None:
    """Display formatted info message"""
    st.markdown(f'<div class="info-message">{message}</div>', unsafe_allow_html=True)

# Load and prepare data with enhanced error handling
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Load happiness data and assign regions through clustering.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with region assignments or None if error occurs
    """
    try:
        if not os.path.exists("happy.csv"):
            display_error("Data file 'happy.csv' not found. Please ensure the file exists in the application directory.")
            return None
        
        df = pd.read_csv("happy.csv")
        
        # Validate required columns
        features = ['gdp', 'social_support', 'life_expectancy', 
                   'freedom_to_make_life_choices', 'generosity', 'corruption']
        missing_columns = [col for col in features if col not in df.columns]
        
        if missing_columns:
            display_error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            return None
        
        # Handle missing values if any
        if df[features].isnull().any().any():
            display_info(f"Dataset contains missing values. They will be filled with median values for clustering.")
            for col in features:
                df[col] = df[col].fillna(df[col].median())
        
        # Standardize features for better clustering
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[features])
            
            # Determine optimal number of clusters (between 3-7)
            optimal_clusters = determine_optimal_clusters(X, range(3, 8))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            df['region_cluster'] = kmeans.fit_predict(X)
            
            # Map cluster numbers to region names
            region_names = {
                0: 'Prosperous Region',
                1: 'Developing Region',
                2: 'Transitional Region',
                3: 'Struggling Region',
                4: 'Emerging Region',
                5: 'High Potential Region',
                6: 'Mixed Progress Region'
            }
            
            # Only use keys that exist in our clustering
            valid_regions = {k: v for k, v in region_names.items() if k < optimal_clusters}
            df['region'] = df['region_cluster'].map(valid_regions)
            
            return df
            
        except Exception as e:
            display_error(f"Error during clustering: {str(e)}")
            # Fallback to simple regional division if clustering fails
            display_info("Using fallback region assignment based on happiness scores.")
            
            # Create simple regions based on happiness score quantiles
            df['happiness_quantile'] = pd.qcut(df['happiness'], 5, labels=False)
            region_fallback = {
                0: 'Lowest Happiness Region',
                1: 'Below Average Region',
                2: 'Average Happiness Region',
                3: 'Above Average Region',
                4: 'Highest Happiness Region'
            }
            df['region'] = df['happiness_quantile'].map(region_fallback)
            return df
            
    except Exception as e:
        display_error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

def determine_optimal_clusters(X: np.ndarray, cluster_range: range) -> int:
    """
    Determine optimal number of clusters using the elbow method.
    
    Args:
        X: Standardized feature matrix
        cluster_range: Range of cluster numbers to try
        
    Returns:
        int: Optimal number of clusters
    """
    try:
        inertia = []
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        # Simple elbow detection - find where the rate of decrease slows down
        inertia_diff = np.diff(inertia)
        inertia_diff_rate = np.diff(inertia_diff)
        
        # Find where the second derivative is maximum (the elbow)
        optimal_idx = np.argmin(inertia_diff_rate) + 1
        return cluster_range[optimal_idx]
    except Exception:
        # Default to 5 clusters if elbow detection fails
        return 5

def create_region_metrics(df: pd.DataFrame) -> None:
    """
    Create and display region metrics.
    
    Args:
        df: DataFrame containing happiness data with regions
    """
    try:
        # Calculate regional statistics
        region_stats = df.groupby('region').agg({
            'happiness': ['mean', 'min', 'max', 'count'],
            'gdp': 'mean',
            'social_support': 'mean',
            'life_expectancy': 'mean',
            'freedom_to_make_life_choices': 'mean',
            'generosity': 'mean',
            'corruption': 'mean'
        })
        
        # Flatten the column hierarchy
        region_stats.columns = ['_'.join(col).strip() for col in region_stats.columns.values]
        region_stats = region_stats.reset_index()
        region_stats = region_stats.round(3)
        
        # Sort by average happiness
        region_stats = region_stats.sort_values('happiness_mean', ascending=False)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Regions", len(region_stats))
        
        with col2:
            happiest_region = region_stats.iloc[0]['region']
            happiest_score = region_stats.iloc[0]['happiness_mean']
            st.metric("Happiest Region", happiest_region, f"Score: {happiest_score:.2f}")
        
        with col3:
            region_gap = region_stats.iloc[0]['happiness_mean'] - region_stats.iloc[-1]['happiness_mean']
            st.metric("Regional Happiness Gap", f"{region_gap:.2f}", "Between highest and lowest")
        
        # Regional happiness bar chart
        fig_region = px.bar(
            region_stats,
            x='region',
            y='happiness_mean',
            color='happiness_mean',
            error_y=region_stats['happiness_mean'] - region_stats['happiness_min'],
            error_y_minus=region_stats['happiness_max'] - region_stats['happiness_mean'],
            title="Average Happiness Score by Region",
            labels={
                'happiness_mean': 'Average Happiness',
                'region': 'Region'
            },
            height=500
        )
        
        fig_region.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Show region statistics table
        with st.expander("View detailed region statistics"):
            display_columns = ['region', 'happiness_mean', 'happiness_min', 'happiness_max', 
                              'happiness_count', 'gdp_mean', 'social_support_mean', 'life_expectancy_mean']
            
            renamed_columns = {
                'region': 'Region',
                'happiness_mean': 'Avg Happiness',
                'happiness_min': 'Min Happiness',
                'happiness_max': 'Max Happiness',
                'happiness_count': 'Countries',
                'gdp_mean': 'Avg GDP',
                'social_support_mean': 'Avg Social Support',
                'life_expectancy_mean': 'Avg Life Expectancy'
            }
            
            display_df = region_stats[display_columns].rename(columns=renamed_columns)
            st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        display_error(f"Error creating region metrics: {str(e)}")

def create_region_comparison_chart(df: pd.DataFrame, selected_factor: str) -> None:
    """
    Create and display region comparison chart for selected factor.
    
    Args:
        df: DataFrame containing happiness data with regions
        selected_factor: Factor to compare across regions
    """
    try:
        # Create box plot for factor distribution by region
        fig_box = px.box(
            df,
            x='region',
            y=selected_factor,
            color='region',
            title=f"{selected_factor.replace('_', ' ').title()} Distribution by Region",
            labels={
                selected_factor: selected_factor.replace('_', ' ').title(),
                'region': 'Region'
            },
            height=500
        )
        
        fig_box.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Add violin plot as alternative view
        show_violin = st.checkbox("Show violin plot instead of box plot")
        
        if show_violin:
            fig_violin = px.violin(
                df,
                x='region',
                y=selected_factor,
                color='region',
                box=True,
                points="all",
                title=f"{selected_factor.replace('_', ' ').title()} Distribution by Region (Violin Plot)",
                labels={
                    selected_factor: selected_factor.replace('_', ' ').title(),
                    'region': 'Region'
                },
                height=500
            )
            
            fig_violin.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
            
    except Exception as e:
        display_error(f"Error creating region comparison chart: {str(e)}")

def create_region_radar_chart(df: pd.DataFrame, selected_region: str) -> None:
    """
    Create and display radar chart for a selected region.
    
    Args:
        df: DataFrame containing happiness data with regions
        selected_region: Region to create radar chart for
    """
    try:
        # Filter data for selected region
        region_data = df[df['region'] == selected_region]
        
        if region_data.empty:
            display_info(f"No data available for {selected_region}")
            return
            
        # Get all regions for comparison
        all_regions = df['region'].unique()
        
        # Features to include in radar chart
        features = ['gdp', 'social_support', 'life_expectancy', 
                   'freedom_to_make_life_choices', 'generosity', 'corruption']
        
        # Calculate average values for each feature by region
        region_avgs = df.groupby('region')[features].mean().reset_index()
        
        # Get global max values for normalization
        max_values = df[features].max()
        
        # Create radar chart
        fig_radar = go.Figure()
        
        # Add trace for selected region
        selected_region_data = region_avgs[region_avgs['region'] == selected_region]
        
        if not selected_region_data.empty:
            # Normalize values
            normalized_values = (selected_region_data[features].values.flatten() / max_values).tolist()
            
            # Close the loop for radar chart
            values = normalized_values + [normalized_values[0]]
            labels = [f.replace('_', ' ').title() for f in features]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                name=selected_region,
                fill='toself',
                line=dict(width=3)
            ))
            
        # Add traces for other regions (transparent)
        for region in all_regions:
            if region != selected_region:
                region_data = region_avgs[region_avgs['region'] == region]
                
                if not region_data.empty:
                    # Normalize values
                    normalized_values = (region_data[features].values.flatten() / max_values).tolist()
                    
                    # Close the loop for radar chart
                    values = normalized_values + [normalized_values[0]]
                    labels = [f.replace('_', ' ').title() for f in features]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels + [labels[0]],
                        name=region,
                        fill='toself',
                        opacity=0.2
                    ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
            title=f"Factor Profile for {selected_region} vs Other Regions",
            legend=dict(x=0.05, y=1.1, orientation='h')
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Add explanation
        st.markdown("""
            <div class="info-message">
                <strong>Understanding the Radar Chart:</strong>
                <p>This chart shows how factors compare across regions. Values are normalized to the global maximum.</p>
                <p>The selected region is shown with a solid line, while other regions are transparent for comparison.</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        display_error(f"Error creating radar chart: {str(e)}")

def create_top_countries_table(df: pd.DataFrame, selected_region: str) -> None:
    """
    Create and display table of top countries in a region.
    
    Args:
        df: DataFrame containing happiness data with regions
        selected_region: Region to show top countries for
    """
    try:
        # Filter data for selected region
        region_data = df[df['region'] == selected_region]
        
        if region_data.empty:
            display_info(f"No data available for {selected_region}")
            return
        
        # Get top countries by happiness
        top_n = min(10, len(region_data))
        top_countries = region_data.nlargest(top_n, 'happiness')[['country', 'happiness', 'gdp', 'life_expectancy']]
        
        # Format columns and add rank
        top_countries = top_countries.reset_index(drop=True)
        top_countries.index = top_countries.index + 1
        top_countries = top_countries.rename(columns={
            'country': 'Country',
            'happiness': 'Happiness Score',
            'gdp': 'GDP per Capita',
            'life_expectancy': 'Life Expectancy'
        })
        
        # Show table
        st.dataframe(top_countries, use_container_width=True)
        
        # Add download option
        csv = region_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {selected_region} Data",
            data=csv,
            file_name=f"{selected_region.replace(' ', '_')}_data.csv",
            mime="text/csv",
        )
        
    except Exception as e:
        display_error(f"Error creating top countries table: {str(e)}")

def main():
    """Main application function"""
    st.title("🌎 Regional Analysis")
    st.write("Explore happiness patterns across different regions of the world")
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
            
        # Show data summary
        with st.expander("Dataset Summary"):
            st.write(f"Total countries: {len(df)}")
            st.write(f"Total regions: {len(df['region'].unique())}")
            
            # Show regions and the number of countries in each
            region_counts = df['region'].value_counts().reset_index()
            region_counts.columns = ['Region', 'Number of Countries']
            st.dataframe(region_counts, use_container_width=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Regional Overview", "Region Comparison", "Region Details"])
        
        with tab1:
            st.subheader("Regional Happiness Overview")
            create_region_metrics(df)
        
        with tab2:
            st.subheader("Cross-Regional Factor Comparison")
            # Select factor for comparison
            factor_options = [
                ('Happiness', 'happiness'),
                ('GDP per Capita', 'gdp'),
                ('Social Support', 'social_support'),
                ('Life Expectancy', 'life_expectancy'),
                ('Freedom', 'freedom_to_make_life_choices'),
                ('Generosity', 'generosity'),
                ('Corruption', 'corruption')
            ]
            
            factor_dict = {name: col for name, col in factor_options}
            selected_factor_name = st.selectbox(
                "Select factor to compare across regions",
                [name for name, _ in factor_options]
            )
            
            # Get corresponding column name
            selected_factor = factor_dict[selected_factor_name]
            
            # Create comparison chart
            create_region_comparison_chart(df, selected_factor)
        
        with tab3:
            st.subheader("Region Details")
            
            # Region selector
            selected_region = st.selectbox("Select Region for Detailed Analysis", sorted(df['region'].unique()))
            
            # Display region metrics
            region_data = df[df['region'] == selected_region]
            if not region_data.empty:
                # Region metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Average Happiness", 
                        f"{region_data['happiness'].mean():.2f}",
                        f"{region_data['happiness'].mean() - df['happiness'].mean():.2f} vs Global"
                    )
                with col2:
                    st.metric(
                        "Countries in Region", 
                        len(region_data),
                        f"{(len(region_data) / len(df) * 100):.1f}% of total"
                    )
                with col3:
                    st.metric(
                        "Happiness Range", 
                        f"{region_data['happiness'].min():.1f} - {region_data['happiness'].max():.1f}",
                        f"Spread: {region_data['happiness'].max() - region_data['happiness'].min():.2f}"
                    )
                
                # Create radar chart
                st.subheader(f"Factor Profile for {selected_region}")
                create_region_radar_chart(df, selected_region)
                
                # Show top countries in region
                st.subheader(f"Top Countries in {selected_region}")
                create_top_countries_table(df, selected_region)
                
                # Show map of countries in region
                st.subheader(f"Countries in {selected_region}")
                
                try:
                    fig_map = px.choropleth(
                        region_data,
                        locations='country',
                        locationmode='country names',
                        color='happiness',
                        hover_name='country',
                        color_continuous_scale='Viridis',
                        title=f'Countries in {selected_region}'
                    )
                    fig_map.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        coloraxis_colorbar_title='Happiness Score'
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create map visualization: {str(e)}")
                    st.info("Some country names may not be recognized by the mapping library.")
                    
                    # Show countries as a list instead
                    countries_list = ', '.join(sorted(region_data['country'].tolist()))
                    st.text_area("Countries in this region:", countries_list, height=100)

    except Exception as e:
        display_error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())

    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built using Streamlit and Plotly</p>
            <p>Data source: World Happiness Report</p>
            <p>Last updated: April 2025</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()