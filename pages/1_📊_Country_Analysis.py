import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import traceback
from typing import Optional, Dict, List, Union

st.set_page_config(page_title="Country Analysis", page_icon="📊", layout="wide")

# Custom CSS for better error messages and UI
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

# Load data with better error handling
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Load and validate the happiness data CSV file.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame containing happiness data or None if error occurs
    """
    try:
        if not os.path.exists("happy.csv"):
            st.error("Data file 'happy.csv' not found. Please ensure the file exists in the application directory.")
            return None
        
        df = pd.read_csv("happy.csv")
        
        # Validate required columns exist
        required_columns = ['country', 'happiness', 'gdp', 'social_support', 'life_expectancy', 
                           'freedom_to_make_life_choices', 'generosity', 'corruption']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            return None
            
        # Convert numeric columns to appropriate types
        numeric_cols = ['happiness', 'gdp', 'social_support', 'life_expectancy', 
                       'freedom_to_make_life_choices', 'generosity', 'corruption']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and handle missing values
        if df.isnull().any().any():
            st.warning(f"Dataset contains {df.isnull().sum().sum()} missing values. They will be handled appropriately.")
            # Fill missing values with medians
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Display error function
def display_error(message: str) -> None:
    """Display formatted error message"""
    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

# Get country data with error handling
def get_country_data(df: pd.DataFrame, country: str) -> Optional[pd.DataFrame]:
    """
    Get data for a specific country with error handling.
    
    Args:
        df: DataFrame containing happiness data
        country: Name of country to filter for
        
    Returns:
        Optional[pd.DataFrame]: Filtered DataFrame or None if country not found
    """
    try:
        country_data = df[df['country'] == country]
        if country_data.empty:
            display_error(f"No data found for {country}. Please select another country.")
            return None
        return country_data
    except Exception as e:
        display_error(f"Error retrieving data for {country}: {str(e)}")
        return None

# Function to create radar chart
def create_radar_chart(country_data: pd.DataFrame, selected_country: str, factors: List[str]) -> Optional[go.Figure]:
    """
    Create radar chart for selected country and factors.
    
    Args:
        country_data: DataFrame containing data for a specific country
        selected_country: Name of selected country
        factors: List of factors to include in the radar chart
        
    Returns:
        Optional[go.Figure]: Plotly figure or None if error occurs
    """
    try:
        fig_radar = go.Figure()
        
        # Normalize values for better visualization
        max_values = {factor: 1.5 * country_data[factor].values[0] for factor in factors}
        values = [country_data[factor].values[0] for factor in factors]
        
        # Make it a closed loop
        values.append(values[0])
        factor_labels = [f.replace('_', ' ').title() for f in factors]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=factor_labels + [factor_labels[0]],
            fill='toself',
            name=selected_country
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.2])),
            showlegend=False,
            title=f"Factor Analysis for {selected_country}"
        )
        
        return fig_radar
    except Exception as e:
        display_error(f"Error creating radar chart: {str(e)}")
        return None

# Function to create comparison chart
def create_comparison_chart(df: pd.DataFrame, selected_country: str, comparison_countries: List[str], 
                           factors: List[str]) -> Optional[go.Figure]:
    """
    Create bar chart comparing countries across factors.
    
    Args:
        df: DataFrame containing happiness data
        selected_country: Main country for comparison
        comparison_countries: List of countries to compare with
        factors: List of factors to include in the comparison
        
    Returns:
        Optional[go.Figure]: Plotly figure or None if error occurs
    """
    try:
        all_countries = [selected_country] + comparison_countries
        comparison_df = df[df['country'].isin(all_countries)]
        
        if comparison_df.empty:
            return None
            
        fig_comparison = go.Figure()
        
        for factor in factors:
            fig_comparison.add_trace(go.Bar(
                name=factor.replace('_', ' ').title(),
                x=comparison_df['country'],
                y=comparison_df[factor],
                text=comparison_df[factor].round(2),
                textposition='auto',
            ))

        fig_comparison.update_layout(
            barmode='group',
            title="Factor Comparison Across Countries",
            xaxis_title="Country",
            yaxis_title="Factor Value"
        )
        
        return fig_comparison
    except Exception as e:
        display_error(f"Error creating comparison chart: {str(e)}")
        return None

# Main function
def main():
    """Main application function"""
    st.title("📊 Country-wise Analysis")
    st.write("Explore happiness factors for individual countries and compare them with others.")
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
            
        # Define factors
        factors = ['gdp', 'social_support', 'life_expectancy', 
                  'freedom_to_make_life_choices', 'generosity', 'corruption']
        factor_labels = [f.replace('_', ' ').title() for f in factors]
        
        # Create sidebar for selections
        with st.sidebar:
            st.header("Country Selection")
            
            # Country selector with search functionality
            countries = sorted(df['country'].unique())
            default_index = countries.index('Niger') if 'Niger' in countries else 0
            
            selected_country = st.selectbox(
                "Select a Country", 
                countries,
                index=default_index,
                help="Choose a country to analyze"
            )
            
            # Let user choose factors to display
            st.header("Factor Selection")
            selected_factors = st.multiselect(
                "Select factors to display",
                factors,
                default=factors[:4],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Choose which happiness factors to include in the analysis"
            )
            
            if not selected_factors:
                selected_factors = factors  # Use all factors if none selected
        
        # Get country data
        country_data = get_country_data(df, selected_country)
        if country_data is None:
            return
            
        # Country metrics in a flexible grid
        st.subheader(f"Happiness Metrics for {selected_country}")
        
        # Use columns for metrics
        cols = st.columns(3)
        
        # Add happiness score with visual indicator
        happiness_score = country_data['happiness'].values[0]
        happiness_rank = df['happiness'].rank(ascending=False)[country_data.index[0]]
        
        cols[0].metric(
            "Happiness Score", 
            f"{happiness_score:.2f}",
            delta=f"Rank: {int(happiness_rank)} of {len(df)}"
        )
        
        # Add GDP per capita with global comparison
        gdp_value = country_data['gdp'].values[0]
        gdp_percentile = (df['gdp'] <= gdp_value).mean() * 100
        
        cols[1].metric(
            "GDP per Capita", 
            f"{gdp_value:.2f}",
            delta=f"Top {100-gdp_percentile:.1f}%"
        )
        
        # Add life expectancy
        life_exp = country_data['life_expectancy'].values[0]
        global_avg = df['life_expectancy'].mean()
        life_delta = life_exp - global_avg
        
        cols[2].metric(
            "Life Expectancy", 
            f"{life_exp:.2f}",
            delta=f"{life_delta:.2f} vs Global Avg"
        )
        
        # Show all factors in a table
        st.subheader("All Happiness Factors")
        
        # Create data for the factors table
        factor_data = {
            "Factor": factor_labels,
            "Value": [country_data[f].values[0] for f in factors],
            "Global Average": [df[f].mean() for f in factors],
            "Global Rank": [int(df[f].rank(ascending=False)[country_data.index[0]]) for f in factors]
        }
        
        factor_df = pd.DataFrame(factor_data)
        st.dataframe(factor_df.style.highlight_max(subset=['Value'], color='lightgreen'))
        
        # Radar chart of factors
        st.subheader("Factor Profile")
        
        # Use selected factors or default to all
        display_factors = selected_factors if selected_factors else factors
        
        # Create and display radar chart
        fig_radar = create_radar_chart(country_data, selected_country, display_factors)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Country comparison
        st.subheader("Compare with Other Countries")
        
        # Allow comparison with up to 5 countries for better visualization
        comparison_countries = st.multiselect(
            "Select countries to compare with",
            [c for c in df['country'].unique() if c != selected_country],
            max_selections=5,
            help="Choose up to 5 countries to compare with the selected country"
        )
        
        if comparison_countries:
            # Create and display comparison chart
            fig_comparison = create_comparison_chart(df, selected_country, comparison_countries, display_factors)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Add download button for comparison data
                comparison_df = df[df['country'].isin([selected_country] + comparison_countries)]
                
                csv = comparison_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Comparison Data",
                    data=csv,
                    file_name=f"{selected_country}_comparison.csv",
                    mime="text/csv",
                )
        else:
            st.info("Select countries to compare with " + selected_country)
            
    except Exception as e:
        display_error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
    