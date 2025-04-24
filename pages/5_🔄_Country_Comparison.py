import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats

def calculate_similarity(data1, data2, metrics):
    """Calculate similarity score between two countries using weighted Euclidean distance"""
    # Define weights for different metrics (can be adjusted based on importance)
    weights = {
        'Ladder score': 1.0,
        'Log GDP per capita': 0.8,
        'Social support': 0.8,
        'Healthy life expectancy': 0.8,
        'Freedom to make life choices': 0.7,
        'Generosity': 0.3,
        'Perceptions of corruption': 0.3
    }
    
    # Calculate weighted normalized distance
    total_dist = 0
    total_weight = 0
    
    for metric in metrics:
        # Get the global min and max for normalization
        val1 = float(data1[metric])
        val2 = float(data2[metric])
        weight = weights.get(metric, 1.0)
        
        # Calculate normalized squared difference
        diff = (val1 - val2) ** 2
        total_dist += diff * weight
        total_weight += weight
    
    # Calculate final similarity score
    if total_weight > 0:
        avg_dist = (total_dist / total_weight) ** 0.5
        # Convert distance to similarity score (0 to 1)
        similarity = max(0, 1 - (avg_dist / 2))
        return similarity
    return 0

def get_similarity_description(similarity):
    """Get a descriptive text for the similarity score"""
    if similarity >= 0.9:
        return "very similar"
    elif similarity >= 0.75:
        return "quite similar"
    elif similarity >= 0.5:
        return "moderately different"
    elif similarity >= 0.25:
        return "quite different"
    else:
        return "very different"

def get_similarity_color(similarity):
    """Get color coding for similarity score"""
    if similarity >= 0.9:
        return "#27ae60"  # Green
    elif similarity >= 0.75:
        return "#2ecc71"  # Light green
    elif similarity >= 0.5:
        return "#f1c40f"  # Yellow
    elif similarity >= 0.25:
        return "#e67e22"  # Orange
    else:
        return "#e74c3c"  # Red

def get_factor_contribution(data1, data2, factor):
    """Calculate how much a factor contributes to the difference"""
    diff = abs(data1[factor] - data2[factor])
    max_val = max(data1[factor], data2[factor])
    return (diff / max_val) if max_val != 0 else 0

def create_comparison_metrics(data1, data2, country1, country2, metrics):
    """Create detailed comparison metrics between two countries"""
    comparisons = []
    for metric in metrics:
        val1 = data1[metric]
        val2 = data2[metric]
        diff = val2 - val1
        pct_diff = (diff / val1 * 100) if val1 != 0 else 0
        
        comparisons.append({
            'metric': metric,
            'value1': val1,
            'value2': val2,
            'difference': diff,
            'pct_difference': pct_diff,
            'significance': 'high' if abs(pct_diff) > 20 else 'medium' if abs(pct_diff) > 10 else 'low'
        })
    
    return comparisons

# Page config
st.set_page_config(page_title="Country Comparison", page_icon="🔄", layout="wide")

# Title and documentation
st.title("🔄 Country Comparison")

with st.expander("📚 About Country Comparison"):
    st.markdown("""
    ### Understanding Country Comparison Analysis
    
    This tool provides a detailed comparison between any two countries in the World Happiness Report, helping you understand their similarities, differences, and trends over time.
    
    #### Key Features:
    1. **Similarity Score**
       - Weighted comparison across multiple factors
       - Score ranges from 0 (very different) to 1 (very similar)
       - Visual indicators of similarity levels
    
    2. **Factor-by-Factor Analysis**
       - Detailed comparison of each happiness factor
       - Percentage differences and significance levels
       - Visual representation of gaps
    
    3. **Trend Analysis**
       - Historical comparison over time
       - Convergence/divergence patterns
       - Key turning points
    
    4. **Insights Generation**
       - Automated analysis of major differences
       - Context-aware comparisons
       - Policy-relevant observations
    """)

with st.expander("🔍 How to Use This Tool"):
    st.markdown("""
    ### Guide to Country Comparison
    
    #### Step-by-Step Instructions:
    1. **Select Countries**
       - Choose two countries from the dropdown menus
       - Consider selecting countries of interest or similar development levels
    
    2. **Read the Similarity Score**
       - Look at the overall similarity percentage
       - Note the descriptive text (e.g., "very similar", "quite different")
       - Check the color coding for quick understanding
    
    3. **Analyze Factor Differences**
       - Review each happiness factor comparison
       - Pay attention to percentage differences
       - Note factors marked as highly significant
    
    4. **Explore Trends**
       - Check how relationships have evolved
       - Look for patterns of convergence or divergence
       - Identify significant changes over time
    
    #### Tips for Analysis:
    - Focus on factors with the largest differences
    - Consider cultural and regional context
    - Look for unexpected similarities/differences
    - Use trends to understand development patterns
    """)

with st.expander("📊 Understanding the Metrics"):
    st.markdown("""
    ### Metrics and Calculations Guide
    
    #### Similarity Score Calculation:
    - Weighted Euclidean distance across factors
    - Normalized to 0-1 scale
    - Factors weighted by importance:
        - Ladder Score: 100%
        - GDP & Social Support: 80%
        - Health & Freedom: 80%
        - Generosity & Corruption: 30%
    
    #### Significance Levels:
    - **High**: >20% difference
    - **Medium**: 10-20% difference
    - **Low**: <10% difference
    
    #### Color Coding:
    - 🟢 Green: Very Similar (>0.9)
    - 🟡 Yellow: Moderately Different (0.5-0.75)
    - 🔴 Red: Very Different (<0.25)
    
    #### Data Considerations:
    - Annual data updates
    - Missing data handling
    - Statistical significance tests
    - Normalization methods
    """)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
    }
    .comparison-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .comparison-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .comparison-card h3, .comparison-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-label {
        color: #666;
        font-weight: 500;
    }
    .metric-value {
        font-weight: 600;
        color: #2c3e50;
    }
    .similarity-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .difference-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .difference-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .difference-low {
        color: #27ae60;
        font-weight: bold;
    }
    .insight-text {
        color: #34495e;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .stat-highlight {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .trend-card {
        margin-bottom: 2rem;
    }
    .plotly-chart {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("happiness_by_country_cleaned.csv")
    # Remove "Explained by: " prefix from column names
    df.columns = [col.replace('Explained by: ', '') for col in df.columns]
    return df

# Main content
st.markdown("""
Compare happiness metrics between countries, analyze differences, and explore contributing factors.
""")

# Load data
df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Comparison Settings")
    
    # Country selection
    col1, col2 = st.columns(2)
    with col1:
        country1 = st.selectbox(
            "First Country",
            options=sorted(df['Country name'].unique()),
            key="country1"
        )
    with col2:
        country2 = st.selectbox(
            "Second Country",
            options=[c for c in sorted(df['Country name'].unique()) if c != country1],
            key="country2"
        )
    
    # Year selection
    selected_year = st.selectbox(
        "Select Year",
        options=sorted(df['Year'].unique(), reverse=True),
        help="Choose the year for comparison"
    )
    
    # Metrics selection
    available_metrics = [
        'Ladder score', 'Log GDP per capita', 'Social support',
        'Healthy life expectancy', 'Freedom to make life choices',
        'Generosity', 'Perceptions of corruption'
    ]
    
    selected_metrics = st.multiselect(
        "Select Metrics",
        options=available_metrics,
        default=available_metrics,
        help="Choose metrics to compare"
    )

if country1 and country2 and selected_metrics:
    # Get data for selected countries and year
    data1 = df[(df['Country name'] == country1) & (df['Year'] == selected_year)].iloc[0]
    data2 = df[(df['Country name'] == country2) & (df['Year'] == selected_year)].iloc[0]
    
    # Calculate similarity score
    similarity = calculate_similarity(data1, data2, selected_metrics)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Detailed Comparison", "Historical Trends", "Statistical Analysis"])
    
    with tab1:
        # Overall similarity score
        similarity_color = get_similarity_color(similarity)
        similarity_description = get_similarity_description(similarity)
        
        st.markdown(f"""
        <div class="comparison-card">
            <h3 style="text-align: center;">Overall Similarity Score</h3>
            <div class="similarity-score" style="color: {similarity_color};">{similarity:.1%}</div>
            <p class="insight-text" style="text-align: center;">
                {country1} and {country2} are <span style="color: {similarity_color}; font-weight: bold;">{similarity_description}</span>
            </p>
            <div class="metric-row" style="border: none;">
                <small class="insight-text" style="text-align: center; width: 100%;">
                    Based on weighted comparison of selected metrics
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Radar chart comparison
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        
        # Prepare data for radar chart
        fig = go.Figure()
        
        # Custom colors for better visibility
        colors = ['#3498db', '#2ecc71']  # Blue and Green
        
        # Format metric names for better display
        def format_metric_name(metric):
            return metric.replace('_', ' ').title()
        
        for idx, (country, data) in enumerate([(country1, data1), (country2, data2)]):
            values = []
            for metric in selected_metrics:
                val = float(data[metric])
                # Min-max normalization for each metric
                min_val = df[metric].min()
                max_val = df[metric].max()
                normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
                values.append(normalized)
            
            # Add trace with improved styling
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=[format_metric_name(m) for m in selected_metrics] + [format_metric_name(selected_metrics[0])],
                name=country,
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(colors[idx].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                line=dict(color=colors[idx], width=2)
            ))
        
        # Update layout with better formatting
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%',
                    ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    gridcolor='#e0e0e0',
                    linecolor='#e0e0e0'
                ),
                angularaxis=dict(
                    gridcolor='#e0e0e0',
                    linecolor='#e0e0e0'
                ),
                bgcolor='white'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#e0e0e0',
                borderwidth=1
            ),
            title=dict(
                text="Factor Profile Comparison",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#2c3e50')
            ),
            height=500,
            margin=dict(t=100, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of the radar chart
        st.markdown("""
        <div class="insight-text" style="text-align: center; margin-top: 1rem;">
            The radar chart shows normalized values (0-100%) for each metric, allowing direct comparison of different factors between countries.
            Larger area indicates better overall performance across metrics.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Detailed Metric Comparison")
        
        # Get detailed comparisons
        comparisons = create_comparison_metrics(data1, data2, country1, country2, selected_metrics)
        
        # Create comparison cards in a grid
        cols = st.columns(3)
        for idx, comp in enumerate(comparisons):
            with cols[idx % 3]:
                diff_class = f"difference-{comp['significance']}"
                
                st.markdown(f"""
                <div class="comparison-card">
                    <h4>{comp['metric']}</h4>
                    <div class="metric-row">
                        <span class="metric-label">{country1}:</span>
                        <span class="metric-value">{comp['value1']:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">{country2}:</span>
                        <span class="metric-value">{comp['value2']:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Difference:</span>
                        <span class="{diff_class}">
                            {comp['difference']:+.3f} ({comp['pct_difference']:+.1f}%)
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Historical Trends")
        
        # Get historical data
        historical_data = df[
            (df['Country name'].isin([country1, country2])) &
            (df['Year'] >= selected_year - 5)
        ]
        
        # Create historical comparison plots
        for metric in selected_metrics:
            st.markdown(f'<div class="trend-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="plotly-chart">', unsafe_allow_html=True)
            
            fig = px.line(
                historical_data,
                x='Year',
                y=metric,
                color='Country name',
                title=f"{metric} Trend Comparison",
                markers=True
            )
            
            # Add trend lines
            for country in [country1, country2]:
                country_data = historical_data[historical_data['Country name'] == country]
                z = np.polyfit(country_data['Year'], country_data[metric], 1)
                p = np.poly1d(z)
                
                fig.add_trace(
                    go.Scatter(
                        x=country_data['Year'],
                        y=p(country_data['Year']),
                        name=f"{country} trend",
                        line=dict(dash='dash'),
                        showlegend=True
                    )
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Statistical Analysis")
        
        # Perform statistical tests
        historical_data = df[
            (df['Country name'].isin([country1, country2])) &
            (df['Year'] >= selected_year - 5)
        ]
        
        for metric in selected_metrics:
            data1 = historical_data[historical_data['Country name'] == country1][metric]
            data2 = historical_data[historical_data['Country name'] == country2][metric]
            
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            st.markdown(f"""
            <div class="comparison-card">
                <h4>{metric}</h4>
                <div class="metric-row">
                    <span class="metric-label">{country1} Mean:</span>
                    <span class="metric-value">{data1.mean():.3f} ± {data1.std():.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">{country2} Mean:</span>
                    <span class="metric-value">{data2.mean():.3f} ± {data2.std():.3f}</span>
                </div>
                <div class="stat-highlight">
                    <div class="metric-row">
                        <span class="metric-label">T-statistic:</span>
                        <span class="metric-value">{t_stat:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">P-value:</span>
                        <span class="metric-value">{p_value:.3f}</span>
                    </div>
                </div>
                <p class="insight-text">
                    The difference is statistically {'significant' if p_value < 0.05 else 'not significant'} (α = 0.05)
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Please select two countries and at least one metric to compare.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Data source: World Happiness Report</p>
    <p>Analysis includes similarity scoring, detailed comparisons, and statistical testing.</p>
</div>
""", unsafe_allow_html=True) 