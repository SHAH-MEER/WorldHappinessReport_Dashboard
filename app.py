import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import numpy as np
import os
import traceback
import json

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
    /* Common styles */
    .main {
        padding: 2rem;
    }
    
    /* Base styles */
    :root {
        --bg-color: #ffffff;
        --text-color: #2c3e50;
        --card-bg: #ffffff;
        --card-border: #e0e0e0;
        --hover-shadow: rgba(0,0,0,0.1);
        --secondary-bg: #f8f9fa;
        --metric-color: #34495e;
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --info-color: #3498db;
    }
    
    .stTitle {
        color: var(--text-color) !important;
        font-size: 3rem !important;
    }
    
    .stSubheader {
        color: var(--metric-color) !important;
        font-size: 1.5rem !important;
    }
    
    .content-card {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--text-color) !important;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px var(--hover-shadow);
        transition: transform 0.2s ease-in-out;
    }
    
    .content-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--hover-shadow);
    }
    
    .content-card h3, .content-card h4 {
        color: var(--text-color) !important;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .content-card p {
        color: var(--text-color) !important;
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    
    .error-message {
        color: var(--danger-color);
        padding: 1rem;
        background-color: var(--card-bg);
        border: 1px solid var(--danger-color);
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .info-message {
        color: var(--info-color);
        padding: 1rem;
        background-color: var(--card-bg);
        border: 1px solid var(--info-color);
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .metric-row {
        background-color: var(--secondary-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .metric-label {
        color: var(--text-color);
        font-weight: 500;
    }
    
    .metric-value {
        color: var(--metric-color);
        font-weight: 600;
    }
    
    /* Style Streamlit native elements */
    .stSelectbox label, .stSlider label {
        color: var(--text-color) !important;
    }
    
    .stMarkdown {
        color: var(--text-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data with better error handling
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data/processed/happiness_by_country_cleaned.csv"):
            st.error("Data file 'happiness_by_country_cleaned.csv' not found. Please ensure the file exists in the application directory.")
            return None
        
        df = pd.read_csv("data/processed/happiness_by_country_cleaned.csv")
        
        # Rename columns to remove "Explained by: " prefix
        df.columns = [col.replace('Explained by: ', '') for col in df.columns]
        
        # Validate required columns exist
        required_columns = ['Country name', 'Ladder score', 'Log GDP per capita', 
                          'Social support', 'Healthy life expectancy', 
                          'Freedom to make life choices', 'Generosity', 
                          'Perceptions of corruption']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            return None
            
        # Ensure data types are correct
        numeric_cols = ['Ladder score', 'Log GDP per capita', 
                       'Social support', 'Healthy life expectancy', 
                       'Freedom to make life choices', 'Generosity', 
                       'Perceptions of corruption']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and handle missing values
        if df.isnull().any().any():
            # Fill missing values with medians for numeric columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

@st.cache_data
def load_latest_data():
    return pd.read_csv("data/processed/latest_happiness_cleaned.csv")

@st.cache_data
def load_summary():
    with open("data/processed/happiness_summary_detailed.txt", "r") as f:
        return f.read()

# Main application content
def main():
    # Title and description with enhanced styling
    st.title("🌍 World Happiness Index")
    
    # Add comprehensive documentation
    with st.expander("📚 About This Dashboard"):
        st.markdown("""
        ### World Happiness Dashboard Guide
        
        This interactive dashboard provides comprehensive insights into global happiness trends and factors affecting well-being across nations.
        
        #### Dashboard Sections:
        1. **Global Overview**: Key metrics and trends
        2. **Happiness Map**: Geographic distribution
        3. **Factor Analysis**: Detailed breakdown of happiness components
        4. **Country Rankings**: Top and bottom performers
        
        #### Data Sources:
        - World Happiness Report data
        - Annual country-level measurements
        - Standardized happiness metrics
        
        #### Key Metrics Explained:
        - **Ladder Score**: Overall happiness rating (0-10)
        - **GDP per capita**: Economic output and living standards
        - **Social Support**: Community and relationship strength
        - **Life Expectancy**: Health and wellness indicators
        - **Freedom**: Personal autonomy measures
        - **Generosity**: Charitable and giving behaviors
        - **Corruption**: Trust in institutions
        """)
    
    st.markdown("""
    <div class="content-card">
        <h3 style='margin-top: 0;'>Welcome to the World Happiness Analysis Dashboard</h3>
        <p>Explore global happiness trends, analyze contributing factors, and discover insights about well-being across nations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data first for all visualizations
    df = load_data()
    if df is None:
        st.error("Could not load or process the dataset. Please check the errors above.")
        return

    # Global Metrics Dashboard
    st.subheader("🌐 Global Happiness Overview")
    
    with st.expander("📈 Understanding Global Metrics"):
        st.markdown("""
        ### Global Happiness Metrics Guide
        
        #### Key Indicators
        1. **Global Happiness Score**
           - Average happiness across all countries
           - Year-over-year change indicator
           - Trend analysis
        
        2. **Happiest Country**
           - Current top performer
           - Score comparison
           - Historical context
        
        3. **Global Stability**
           - Measures happiness score volatility
           - Lower values indicate more stable scores
           - Important for trend analysis
        
        #### How to Use These Metrics
        - Monitor global trends
        - Identify significant changes
        - Compare regional performance
        - Track stability over time
        
        #### Data Updates
        - Annual data refreshes
        - Rolling averages for stability
        - Seasonal adjustments where applicable
        """)
    
    # Create metrics with sparklines
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics
    current_year = df['Year'].max()
    prev_year = current_year - 1
    current_avg = df[df['Year'] == current_year]['Ladder score'].mean()
    prev_avg = df[df['Year'] == prev_year]['Ladder score'].mean()
    
    with col1:
        st.metric(
            "Global Happiness",
            f"{current_avg:.2f}",
            f"{((current_avg - prev_avg) / prev_avg) * 100:.1f}%",
            help="Average global happiness score and year-over-year change"
        )

    with col2:
        happiest_country = df[df['Year'] == current_year].nlargest(1, 'Ladder score')['Country name'].iloc[0]
        happiest_score = df[df['Year'] == current_year].nlargest(1, 'Ladder score')['Ladder score'].iloc[0]
        st.metric(
            "Happiest Country",
            happiest_country,
            f"Score: {happiest_score:.2f}"
        )

    with col3:
        volatility = df.groupby('Country name')['Happiness Change'].std().mean()
        st.metric(
            "Global Stability",
            f"{volatility:.3f}",
            "Lower is more stable",
            help="Average happiness score volatility across countries"
        )

    # Happiness Highlights Section
    st.markdown("### 🎯 Happiness Highlights")
    
    with st.expander("🔍 Understanding Happiness Highlights"):
        st.markdown("""
        ### Happiness Highlights Guide
        
        #### Most Improved Countries
        - Shows countries with largest positive changes
        - Based on year-over-year improvement
        - Considers multiple factors
        
        #### Quick Facts
        - Total countries analyzed
        - Data timespan
        - Score distributions
        - Regional patterns
        
        #### Analysis Tips
        1. Look for regional patterns in improvements
        2. Consider economic and social factors
        3. Note stability of improvements
        4. Compare with global averages
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Most Improved Countries
        most_improved = df.groupby('Country name')['Happiness Change'].mean().nlargest(5)
        
        fig_improved = go.Figure()
        fig_improved.add_trace(go.Bar(
            x=most_improved.index,
            y=most_improved.values,
            marker_color='#2ecc71',
            text=[f"+{val:.2f}" for val in most_improved.values],
            textposition='auto',
        ))
        
        fig_improved.update_layout(
            title="Most Improved Countries",
            xaxis_title="Country",
            yaxis_title="Average Annual Improvement",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_improved, use_container_width=True)
    
    with col2:
        # Key Statistics
        st.markdown("""
        <div class="content-card">
            <h4 style='margin-top: 0;'>Quick Facts</h4>
            <ul>
                <li>Data spans {years} years</li>
                <li>{countries} countries analyzed</li>
                <li>Average happiness: {avg:.2f}</li>
                <li>Score range: {min:.1f} - {max:.1f}</li>
            </ul>
        </div>
        """.format(
            years=df['Year'].nunique(),
            countries=df['Country name'].nunique(),
            avg=df['Ladder score'].mean(),
            min=df['Ladder score'].min(),
            max=df['Ladder score'].max()
        ), unsafe_allow_html=True)

    # Interactive World Map
    st.markdown("### 🗺️ Global Happiness Distribution")
    
    with st.expander("🌐 Map Navigation Guide"):
        st.markdown("""
        ### World Map Visualization Guide
        
        #### Map Features
        1. **Color Coding**
           - Darker colors = Higher happiness scores
           - Lighter colors = Lower happiness scores
           - Gray = No data available
        
        2. **Interaction Tools**
           - Hover for detailed information
           - Click and drag to pan
           - Scroll to zoom
           - Double-click to reset view
        
        3. **Data Display**
           - Country name
           - Happiness score
           - GDP per capita
           - Year-over-year change
        
        #### Analysis Tips
        - Look for regional patterns
        - Compare neighboring countries
        - Note development correlations
        - Track changes over time
        
        #### Technical Notes
        - Data updates annually
        - Some countries may lack data
        - Scores normalized for comparison
        """)
    
    # Year selector for map
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        help="Drag to see happiness distribution for different years"
    )
    
    # Filter data for selected year
    year_data = df[df['Year'] == selected_year]
    
    # Create choropleth map
    fig_map = px.choropleth(
        year_data,
        locations='Country name',
        locationmode='country names',
        color='Ladder score',
        hover_name='Country name',
        hover_data={
            'Ladder score': ':.2f',
            'Log GDP per capita': ':.3f',
            'Happiness Change': ':.3f'
        },
        color_continuous_scale='Viridis',
        title=f'World Happiness Scores ({selected_year})'
    )
    
    fig_map.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    

    # Factor Analysis Section
    st.markdown("### 📊 Happiness Factors Analysis")
    
    
    
    # Get factor correlations
    factor_cols = ['Ladder score', 'Log GDP per capita', 
                  'Social support', 'Healthy life expectancy',
                  'Freedom to make life choices', 'Generosity',
                  'Perceptions of corruption']
    
    corr_matrix = df[factor_cols].corr().round(2)
    
    # Factor Impact Analysis
    factor_importance = pd.Series({
        col: abs(corr_matrix['Ladder score'][col])
        for col in corr_matrix.columns if col != 'Ladder score'
    }).sort_values(ascending=True)
    
    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        y=factor_importance.index,
        x=factor_importance.values,
        orientation='h',
        marker_color='#3498db',
        text=[f"{val:.2f}" for val in factor_importance.values],
        textposition='auto',
    ))
    
    fig_impact.update_layout(
        title="Factor Impact on Happiness",
        xaxis_title="Correlation Strength",
        yaxis_title="",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 1]),
        margin=dict(l=0, r=10, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_impact, use_container_width=True)
    
    # Add explanation
    st.info("""
    **Understanding Factor Impact:**
    - This analysis shows how strongly each factor correlates with overall happiness scores
    - Higher values indicate stronger relationships with happiness
    - Values range from 0 (no correlation) to 1 (perfect correlation)
    """)

    # Enhanced sidebar with better organization
    with st.sidebar:
        st.sidebar.header("📊 Dashboard Controls")
        
        # Add country selection
        selected_country = st.multiselect(
            "Select Countries to Compare",
            df['Country name'].unique(),
            help="Choose one or more countries to analyze"
        )
        
        # Add factor selection
        factor_options = [
            ('GDP', 'Log GDP per capita'),
            ('Social Support', 'Social support'),
            ('Life Expectancy', 'Healthy life expectancy'),
            ('Freedom', 'Freedom to make life choices'),
            ('Generosity', 'Generosity'),
            ('Corruption', 'Perceptions of corruption')
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
        col1, col2 = st.columns([2, 1])
        filtered_df = df[df['Country name'].isin(selected_country)] if selected_country else df.copy()
        
        with col1:
            st.subheader("📊 Interactive Scatter Plot")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) >= 2:
                try:
                    x_axis = st.selectbox("X-axis", factor_cols, index=1)  # Default to GDP
                    y_axis = st.selectbox("Y-axis", factor_cols, index=0)  # Default to Ladder score
                    
                    # Create scatter plot with error handling
                    if filtered_df.empty:
                        st.info("No data to display. Please select at least one country or clear the selection.")
                    else:
                        fig_scatter = px.scatter(filtered_df, 
                                            x=x_axis, 
                                            y=y_axis,
                                            color='Country name',
                                            size='Ladder score',
                                            hover_data=['Country name'],
                                            title=f"{x_axis} vs {y_axis}")
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

        # Top 10 happiest countries
        st.subheader("🏆 Top 10 Happiest Countries")
        try:
            top_10 = df[df['Year'] == df['Year'].max()].nlargest(10, 'Ladder score')
            fig_bar = px.bar(
                top_10,
                x='Country name',
                y='Ladder score',
                color='Ladder score',
                title="Top 10 Happiest Countries",
                labels={"Country name": "Country", "Ladder score": "Happiness Score"}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top countries chart: {str(e)}")

        st.subheader("😞 Top 10 Unhappiest Countries")
        try:
            bottom_10 = df[df['Year'] == df['Year'].max()].nsmallest(10, 'Ladder score')
            fig_bar = px.bar(
                bottom_10,
                x='Country name',
                y='Ladder score',
                color='Ladder score',
                title="Top 10 Unhappiest Countries",
                labels={"Country name": "Country", "Ladder score": "Happiness Score"}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating bottom countries chart: {str(e)}")

        # Factors comparison with radar chart
        st.subheader("🔍 Happiness Factors Comparison")
        
        # Only show radar chart if countries are selected
        if selected_country:
            try:
                # Get factor columns or use defaults
                factors = selected_factor_cols if selected_factor_cols else [col for col in factor_cols if col != 'Ladder score']
                
                # Normalize factors for better comparison
                df_normalized = df.copy()
                for factor in factors:
                    df_normalized[factor] = (df[factor] - df[factor].min()) / (df[factor].max() - df[factor].min())
                
                fig_radar = go.Figure()
                
                for country in selected_country:
                    country_data = df_normalized[df_normalized['Country name'] == country].iloc[-1]  # Get latest data
                    
                    if not country_data.empty:
                        # Get normalized values for selected factors
                        values = [country_data[factor] for factor in factors]
                        # Add first value again to close the polygon
                        values.append(values[0])
                        
                        # Create labels with better formatting
                        labels = [factor.replace('Log GDP per capita', 'GDP').replace('_', ' ').title() for factor in factors]
                        labels.append(labels[0])  # Add first label again to close the polygon
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=labels,
                            name=country,
                            fill='toself'
                        ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="Normalized Factor Comparison",
                    height=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Add explanatory text
                st.info("""
                **How to read this chart:**
                - Values are normalized (0-1 scale) for comparison
                - Each axis represents a happiness factor
                - Larger area indicates better overall performance
                - Compare shapes to understand relative strengths
                """)
                
            except Exception as e:
                st.error(f"Error creating radar chart: {str(e)}")
                st.error(traceback.format_exc())
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