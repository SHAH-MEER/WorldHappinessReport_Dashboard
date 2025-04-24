import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import numpy as np
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
    .mapbox-token-warning {
        display: none !important;
    }
    .content-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .content-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .content-card h3, .content-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .content-card p {
        color: #4a4a4a;
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    .content-card ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        list-style-type: none;
    }
    .content-card li {
        color: #4a4a4a;
        margin: 0.5rem 0;
        position: relative;
    }
    .content-card li:before {
        content: "•";
        color: #3498db;
        font-weight: bold;
        position: absolute;
        left: -1rem;
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
    
    # Add map legend and instructions
    with st.expander("🔍 Map Instructions"):
        st.markdown("""
        <div class="content-card">
            <ul>
                <li>Hover over countries to see detailed happiness metrics</li>
                <li>Click and drag to pan across the map</li>
                <li>Scroll to zoom in/out</li>
                <li>Double-click to reset the view</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
                factors = selected_factor_cols if selected_factor_cols else factor_cols[1:]  # Exclude Ladder score
                
                # Normalize factors for better comparison
                max_values = df[factors].max()
                
                fig_radar = go.Figure()
                
                for country in selected_country:
                    country_data = df[df['Country name'] == country]
                    
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