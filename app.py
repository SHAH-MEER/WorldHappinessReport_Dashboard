import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import traceback

# Custom theme and styling
st.set_page_config(
    page_title="World Happiness Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .stSubheader {
        color: #34495e;
        font-size: 1.5rem !important;
    }
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
    </style>
""", unsafe_allow_html=True)

# Load and prepare data with better error handling
@st.cache_data
def load_data():
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
            
        # Ensure data types are correct
        numeric_cols = ['happiness', 'gdp', 'social_support', 'life_expectancy', 
                       'freedom_to_make_life_choices', 'generosity', 'corruption']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and handle missing values
        if df.isnull().any().any():
            st.warning(f"Dataset contains {df.isnull().sum().sum()} missing values. They will be handled appropriately.")
            # Fill missing values with medians for numeric columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Main application content
def main():
    # Title and description with enhanced styling
    st.title("🌍 World Happiness Index")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <h3 style='color: #2c3e50;'>Welcome to the World Happiness Analysis Dashboard</h3>
        <p>Explore global happiness trends, analyze contributing factors, and discover insights about well-being across nations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        st.error("Could not load or process the dataset. Please check the errors above.")
        return

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Enhanced sidebar with better organization
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/happiness.png", width=100)
        st.sidebar.header("📊 Dashboard Controls")
        
        # Add country selection
        selected_country = st.multiselect(
            "Select Countries to Compare",
            df['country'].unique(),
            help="Choose one or more countries to analyze"
        )
        
        # Add factor selection
        factor_options = [
            ('GDP', 'gdp'),
            ('Social Support', 'social_support'),
            ('Life Expectancy', 'life_expectancy'),
            ('Freedom', 'freedom_to_make_life_choices'),
            ('Generosity', 'generosity'),
            ('Corruption', 'corruption')
        ]
        selected_factors = st.multiselect(
            "Select Factors to Analyze",
            [x[0] for x in factor_options],
            default=[x[0] for x in factor_options[:3]],
            help="Choose which happiness factors to include in analysis"
        )
        
        # Convert friendly names back to column names
        factor_dict = {x[0]: x[1] for x in factor_options}
        selected_factor_cols = [factor_dict[factor] for factor in selected_factors] if selected_factors else []
        
        st.sidebar.markdown("---")
        
        # Add download option
        if st.download_button(
            label="Download Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='happiness_data.csv',
            mime='text/csv',
        ):
            st.sidebar.success("Download started!")

    # Main content with improved layout
    try:
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.metric(
                "Global Average Happiness",
                f"{df['happiness'].mean():.2f}",
                f"{df['happiness'].std():.2f} σ"
            )

        with col2:
            st.metric(
                "Happiest Country",
                df.loc[df['happiness'].idxmax(), 'country'],
                f"Score: {df['happiness'].max():.2f}"
            )

        with col3:
            st.metric(
                "Total Countries",
                len(df['country'].unique()),
                "Analyzed"
            )

        # World map visualization with error handling
        st.subheader("🗺️ Global Happiness Distribution")
        try:
            fig_map = px.choropleth(
                df,
                locations='country',
                locationmode='country names',
                color='happiness',
                hover_name='country',
                color_continuous_scale='Magma',
                range_color=[df['happiness'].min(), df['happiness'].max()],
                scope='world',
                labels={'happiness': 'Happiness Score'},
                title='World Happiness Map'
            )
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar_title='Happiness Score'
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating map visualization: {str(e)}")
            st.info("Try selecting different countries or check if your countries have valid names for mapping.")

        # Main layout with columns
        col1, col2 = st.columns([2, 1])
        filtered_df = df[df['country'].isin(selected_country)] if selected_country else df.copy()
        
        with col1:
            st.subheader("📊 Interactive Scatter Plot")
            if len(numeric_columns) >= 2:
                try:
                    x_axis = st.selectbox("X-axis", numeric_columns, index=1)  # Default to gdp
                    y_axis = st.selectbox("Y-axis", numeric_columns, index=0)  # Default to happiness
                    
                    # Create scatter plot with error handling
                    if filtered_df.empty:
                        st.info("No data to display. Please select at least one country or clear the selection.")
                    else:
                        fig_scatter = px.scatter(filtered_df, 
                                            x=x_axis, 
                                            y=y_axis,
                                            color='country',
                                            size='happiness',
                                            hover_data=['country'],
                                            title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
            else:
                st.warning("Not enough numeric columns for a scatter plot.")

        with col2:
            st.subheader("📈 Key Statistics")
            if filtered_df.empty:
                st.info("Select at least one country to see statistics.")
            else:
                st.dataframe(filtered_df.describe().round(2))

        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        try:
            corr = df[numeric_columns].corr().round(2)
            fig_heatmap = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r',
                text_auto=True
            )
            fig_heatmap.update_layout(title="Correlation Between Happiness Factors")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")

        # Top 10 happiest countries
        st.subheader("🏆 Top 10 Happiest Countries")
        try:
            top_10 = df.nlargest(10, 'happiness')
            fig_bar = px.bar(
                top_10,
                x='country',
                y='happiness',
                color='happiness',
                title="Top 10 Happiest Countries",
                labels={"country": "Country", "happiness": "Happiness Score"}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top countries chart: {str(e)}")


        st.subheader("😞 Top 10 Unhappiest Countries")
        try:
            top_10 = df.nsmallest(10, 'happiness')
            fig_bar = px.bar(
                top_10,
                x='country',
                y='happiness',
                color='happiness',
                title="Top 10 Unhappiest Countries",
                labels={"country": "Country", "happiness": "Happiness Score"}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top countries chart: {str(e)}")

        # Factors comparison with radar chart
        st.subheader("🔍 Happiness Factors Comparison")
        
        # Only show radar chart if countries are selected
        if selected_country:
            try:
                # Get factor columns or use defaults
                factors = selected_factor_cols if selected_factor_cols else [
                    'gdp', 'social_support', 'life_expectancy', 
                    'freedom_to_make_life_choices', 'generosity', 'corruption'
                ]
                
                # Normalize factors for better comparison
                max_values = df[factors].max()
                
                fig_radar = go.Figure()
                
                for country in selected_country:
                    country_data = df[df['country'] == country]
                    
                    if not country_data.empty:
                        # Normalize values for better visualization
                        normalized_values = (country_data[factors].values.flatten() / max_values).tolist()
                        
                        # Close the loop for radar chart
                        values = normalized_values + [normalized_values[0]]
                        labels = [f.replace('_', ' ').title() for f in factors]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=labels + [labels[0]],
                            name=country,
                            fill='toself'
                        ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Normalized Factor Comparison"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Add explanatory text
                st.info("Note: Values are normalized relative to the maximum value across all countries to enable comparison.")
                
            except Exception as e:
                st.error(f"Error creating radar chart: {str(e)}")
        else:
            st.info("Select countries to compare their happiness factors.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())

    # Enhanced footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align: center; color: #666;'>
                <p>Built using Streamlit and Plotly</p>
                <p>Data source: World Happiness Report</p>
                <p>Last updated: April 2025</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()