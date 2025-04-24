import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from ruptures import Binseg  # for change point detection

# Page config
st.set_page_config(page_title="Time Series Analysis", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    .metric-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    .metric-card p {
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    .highlight {
        color: #2980b9;
        font-weight: bold;
    }
    .trend-positive {
        color: #27ae60;
        font-weight: bold;
    }
    .trend-negative {
        color: #c0392b;
        font-weight: bold;
    }
    .analysis-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/happiness_by_country_cleaned.csv")
    df.columns = [col.replace('Explained by: ', '') for col in df.columns]
    return df

def calculate_trend(data):
    """Calculate trend statistics for a time series."""
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    trend = "Increasing" if slope > 0 else "Decreasing"
    strength = abs(r_value)
    return trend, strength, slope

def perform_seasonality_analysis(data, column):
    """Perform seasonality analysis on time series data"""
    try:
        # Ensure data is sorted by time
        data = data.sort_values('Year')
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data[column], period=2, model='additive')
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    except Exception as e:
        st.warning(f"Could not perform seasonality analysis: {str(e)}")
        return None

def detect_change_points(data, column):
    """Detect significant changes in time series"""
    try:
        # Prepare data for change point detection
        signal = data[column].values
        algo = Binseg(model="l2").fit(signal.reshape(-1, 1))
        change_points = algo.predict(n_bkps=3)  # Detect top 3 change points
        
        return change_points
    except Exception as e:
        st.warning(f"Could not perform change point detection: {str(e)}")
        return []

# Main content
st.title("📈 Happiness Time Series Analysis")
st.markdown("""
Analyze happiness trends over time, compare country trajectories, and explore historical patterns.
Use the controls below to customize your analysis.
""")

# Documentation and Information Sections
with st.expander("ℹ️ About This Page"):
    st.markdown("""
    The Time Series Analysis page allows you to explore how happiness scores and related metrics have changed over time.
    You can compare multiple countries, analyze trends, and identify patterns in the data.
    
    **Key Features:**
    - Interactive trend visualization with optional trend lines
    - Year-over-year change analysis
    - Comparative analysis with normalized trends
    - Statistical correlation between countries
    - Trend strength and volatility metrics
    """)

with st.expander("📊 Available Analyses"):
    st.markdown("""
    ### 1. Happiness Trends
    - View happiness score trends over time
    - Compare multiple countries simultaneously
    - Add trend lines to identify overall direction
    - See trend statistics including strength and annual change
    
    ### 2. Year-over-Year Changes
    - Analyze annual changes in happiness scores
    - Identify significant improvements or declines
    - Compare volatility between countries
    - View summary statistics of changes
    
    ### 3. Comparative Analysis
    - Compare happiness distributions across countries
    - View normalized trends for better comparison
    - Analyze correlations between country happiness scores
    - Identify similar patterns between countries
    """)

with st.expander("🔍 How to Use"):
    st.markdown("""
    1. **Select Countries:**
       - Use the sidebar to choose countries you want to analyze
       - The default selection shows the top 3 happiest countries
    
    2. **Choose Time Range:**
       - Use the year range slider to focus on specific periods
       - Compare different time periods to identify patterns
    
    3. **Customize Visualization:**
       - Toggle trend lines on/off
       - Enable/disable confidence intervals
       - Switch between different analysis tabs
    
    4. **Interpret Results:**
       - Trend cards show the direction and strength of trends
       - Year-over-year changes indicate stability and volatility
       - Correlation analysis shows relationships between countries
    """)

# Load data
df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Analysis Controls")
    
    # Country selection
    selected_countries = st.multiselect(
        "Select Countries to Compare",
        options=sorted(df['Country name'].unique()),
        default=df.groupby('Country name')['Ladder score'].mean().nlargest(3).index.tolist(),
        help="Choose one or more countries to analyze"
    )
    
    # Time range selection
    year_range = st.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        help="Choose the time period to analyze"
    )
    
    # Trend options
    show_trend = st.checkbox("Show Trend Lines", value=True)
    show_confidence = st.checkbox("Show Confidence Interval", value=False)

# Filter data based on selections
filtered_df = df[
    (df['Country name'].isin(selected_countries)) &
    (df['Year'].between(year_range[0], year_range[1]))
].copy()  # Add .copy() to avoid SettingWithCopyWarning

# Sort the data by Year to ensure proper line plotting
filtered_df = filtered_df.sort_values(['Country name', 'Year'])

# Calculate year-over-year changes for each country
filtered_df['Happiness Change'] = filtered_df.groupby('Country name')['Ladder score'].diff()

# Main content layout
if not selected_countries:
    st.info("Please select at least one country from the sidebar to begin analysis.")
else:
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Happiness Trends", "Year-over-Year Changes", "Comparative Analysis"])
    
    with tab1:
        st.subheader("Happiness Score Trends")
        
        # Main trend plot
        fig = px.line(
            filtered_df,
            x='Year',
            y='Ladder score',
            color='Country name',
            title="Happiness Trends Over Time",
            labels={'Ladder score': 'Happiness Score', 'Year': 'Year', 'Country name': 'Country'},
            markers=True  # Add markers to the lines
        )
        
        # Update layout for better visibility
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',  # Show all years
                dtick=1,  # Space between ticks
                gridcolor='rgba(128, 128, 128, 0.2)',  # Lighter grid
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',  # Lighter grid
                showgrid=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent surrounding
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent legend
                bordercolor="rgba(128, 128, 128, 0.2)",  # Light border
                borderwidth=1
            ),
            margin=dict(r=150),  # Add right margin to accommodate legend
            hovermode='x unified'  # Show all points at same x-coordinate
        )
        
        if show_trend:
            for country in selected_countries:
                country_data = filtered_df[filtered_df['Country name'] == country]
                if not country_data.empty:
                    x = country_data['Year']
                    y = country_data['Ladder score']
                    
                    # Only calculate trend if we have enough data points
                    if len(x) > 1:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        # Add trend line
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=p(x),
                                name=f"{country} trend",
                                line=dict(dash='dash', width=1),
                                showlegend=False
                            )
                        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # After the main time series plot, add metric cards
        cols = st.columns(3)

        # First card: Trend Analysis
        with cols[0]:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Trend Analysis</h4>
                <p>Overall Trend: <span class="highlight">Long-term Pattern</span></p>
                <ul>
                    <li>Start Value: {start_value:.2f}</li>
                    <li>End Value: {end_value:.2f}</li>
                    <li>Net Change: <span class="{trend_class}">{net_change:+.2f}</span></li>
                </ul>
                <p>Trend indicates {trend_description}</p>
            </div>
            """.format(
                start_value=filtered_df.iloc[0]['Ladder score'],
                end_value=filtered_df.iloc[-1]['Ladder score'],
                net_change=filtered_df.iloc[-1]['Ladder score'] - filtered_df.iloc[0]['Ladder score'],
                trend_class='trend-positive' if filtered_df.iloc[-1]['Ladder score'] > filtered_df.iloc[0]['Ladder score'] else 'trend-negative',
                trend_description='improvement' if filtered_df.iloc[-1]['Ladder score'] > filtered_df.iloc[0]['Ladder score'] else 'decline'
            ), unsafe_allow_html=True)

        # Second card: Volatility Analysis
        with cols[1]:
            volatility = filtered_df['Ladder score'].std()
            avg_yearly_change = abs(filtered_df['Ladder score'].diff()).mean()
            
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Volatility Analysis</h4>
                <p>Score Stability Metrics:</p>
                <ul>
                    <li>Standard Deviation: {volatility:.3f}</li>
                    <li>Avg. Yearly Change: {avg_change:.3f}</li>
                    <li>Stability Level: <span class="highlight">{stability}</span></li>
                </ul>
                <p>Based on historical fluctuations</p>
            </div>
            """.format(
                volatility=volatility,
                avg_change=avg_yearly_change,
                stability='High' if volatility < 0.2 else 'Medium' if volatility < 0.5 else 'Low'
            ), unsafe_allow_html=True)

        # Third card: Key Statistics
        with cols[2]:
            peak_year = filtered_df.loc[filtered_df['Ladder score'].idxmax(), 'Year']
            peak_score = filtered_df['Ladder score'].max()
            trough_year = filtered_df.loc[filtered_df['Ladder score'].idxmin(), 'Year']
            trough_score = filtered_df['Ladder score'].min()
            
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Key Statistics</h4>
                <p>Historical Extremes:</p>
                <ul>
                    <li>Peak: <span class="trend-positive">{peak:.2f}</span> ({peak_year})</li>
                    <li>Trough: <span class="trend-negative">{trough:.2f}</span> ({trough_year})</li>
                    <li>Range: {range:.2f}</li>
                </ul>
                <p>Based on selected time period</p>
            </div>
            """.format(
                peak=peak_score,
                peak_year=int(peak_year),
                trough=trough_score,
                trough_year=int(trough_year),
                range=peak_score - trough_score
            ), unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Year-over-Year Changes")
        
        # Year-over-year changes bar plot
        yoy_fig = px.bar(
            filtered_df,
            x='Year',
            y='Happiness Change',
            color='Country name',
            barmode='group',
            title="Year-over-Year Happiness Changes",
            labels={'Happiness Change': 'Change in Happiness', 'Year': 'Year', 'Country name': 'Country'}
        )
        st.plotly_chart(yoy_fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### Change Statistics")
        summary_df = filtered_df.groupby('Country name')['Happiness Change'].agg([
            ('Average Change', 'mean'),
            ('Max Increase', 'max'),
            ('Max Decrease', 'min'),
            ('Volatility', 'std')
        ]).round(3)
        st.dataframe(summary_df)
    
    with tab3:
        st.subheader("Comparative Analysis")
        
        # Create subplot with individual trends and distributions
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Happiness Distribution", "Trend Comparison"),
            specs=[[{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Box plot
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country name'] == country]
            fig.add_trace(
                go.Box(y=country_data['Ladder score'], name=country),
                row=1, col=1
            )
        
        # Normalized trends
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country name'] == country]
            normalized_score = (country_data['Ladder score'] - country_data['Ladder score'].min()) / \
                             (country_data['Ladder score'].max() - country_data['Ladder score'].min())
            
            fig.add_trace(
                go.Scatter(x=country_data['Year'], y=normalized_score, name=country, mode='lines+markers'),
                row=1, col=2
            )
        
        fig.update_layout(height=500, title_text="Comparative Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(selected_countries) > 1:
            st.markdown("### Correlation Analysis")
            pivot_df = filtered_df.pivot(index='Year', columns='Country name', values='Ladder score')
            corr_matrix = pivot_df.corr().round(2)
            
            corr_fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            corr_fig.update_layout(title="Happiness Score Correlations Between Countries")
            st.plotly_chart(corr_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Data source: World Happiness Report</p>
    <p>Time series analysis includes trends, year-over-year changes, and comparative metrics.</p>
</div>
""", unsafe_allow_html=True) 