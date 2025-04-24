import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import zscore

# Page config
st.set_page_config(page_title="Country Profile & Analysis", page_icon="🌎", layout="wide")

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
    .profile-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .trend-indicator {
        font-weight: bold;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        margin-left: 0.5rem;
    }
    .trend-up {
        color: #27ae60;
        background-color: #eafaf1;
    }
    .trend-down {
        color: #e74c3c;
        background-color: #fdf3f2;
    }
    .peer-comparison {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .section-title {
        color: #2c3e50;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .help-text {
        color: #666;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/happiness_by_country_cleaned.csv")
    df.columns = [col.replace('Explained by: ', '') for col in df.columns]
    return df

# Load data
df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Analysis Controls")
    
    # Country selection
    selected_country = st.selectbox(
        "Select a Country",
        options=sorted(df['Country name'].unique()),
        help="Choose a country to analyze"
    )
    
    # Year selection
    selected_year = st.selectbox(
        "Select Year",
        options=sorted(df['Year'].unique(), reverse=True),
        help="Choose the year for analysis"
    )
    
    # Metrics selection
    available_metrics = [
        'Ladder score', 'Log GDP per capita', 'Social support',
        'Healthy life expectancy', 'Freedom to make life choices',
        'Generosity', 'Perceptions of corruption'
    ]
    
    selected_metrics = st.multiselect(
        "Select Factors",
        options=available_metrics,
        default=available_metrics,
        help="Choose which factors to analyze"
    )

# Main content
st.title("🌎 Country Profile & Analysis")
st.markdown("""
Explore detailed country profiles, analyze happiness factors, and understand historical trends.
""")

# Help sections in expanders
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Select a country from the sidebar
    2. Choose the year of analysis
    3. Select factors to analyze
    4. Explore different tabs for various insights:
        - Overview: Quick summary and key metrics
        - Trends: Historical data visualization
        - Factor Analysis: Detailed breakdown of contributing factors
        - Peer Comparison: How the country compares to others
    """)

with st.expander("📊 Understanding the Analysis"):
    st.markdown("""
    - **Ladder Score**: Overall happiness score (0-10)
    - **GDP per capita**: Economic output and living standards
    - **Social support**: Having someone to count on in times of trouble
    - **Healthy life expectancy**: Number of healthy years of life expected
    - **Freedom**: Freedom to make life choices
    - **Generosity**: How generous people are to each other
    - **Corruption**: Perceptions of corruption in government/business
    """)

# Get country data
country_data = df[df['Country name'] == selected_country]
current_data = country_data[country_data['Year'] == selected_year].iloc[0]
previous_year_data = country_data[country_data['Year'] == selected_year - 1]
has_previous_year = not previous_year_data.empty

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Factor Analysis", "Peer Comparison"])

with tab1:
    # Country Profile Card
    st.markdown("""
    <div class="profile-card">
        <h2>🌍 Country Profile</h2>
        <div class="metric-row">
            <span class="metric-label">Happiness Rank</span>
            <span class="metric-value">#{}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Happiness Score</span>
            <span class="metric-value">{:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Region</span>
            <span class="metric-value">{}</span>
        </div>
    </div>
    """.format(
        df[df['Year'] == selected_year]['Ladder score'].rank(ascending=False)[current_data.name].astype(int),
        current_data['Ladder score'],
        "Region"  # You can add region information if available in your dataset
    ), unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("Key Metrics")
    cols = st.columns(3)
    
    for idx, metric in enumerate(selected_metrics[:6]):  # Show up to 6 metrics
        with cols[idx % 3]:
            current_value = current_data[metric]
            if has_previous_year:
                previous_value = previous_year_data.iloc[0][metric]
                change = current_value - previous_value
                change_pct = (change / previous_value * 100) if previous_value != 0 else 0
                trend_class = "trend-up" if change >= 0 else "trend-down"
                trend_symbol = "↑" if change >= 0 else "↓"
            else:
                change_pct = 0
                trend_class = ""
                trend_symbol = ""
            
            st.markdown(f"""
            <div class="profile-card">
                <div class="metric-label">{metric}</div>
                <div class="metric-value">
                    {current_value:.3f}
                    {f'<span class="{trend_class}">{trend_symbol} {abs(change_pct):.1f}%</span>' if has_previous_year else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Historical Trends")
    
    # Get historical data for the country
    historical_data = df[df['Country name'] == selected_country].sort_values('Year')
    
    # Create line chart for happiness score trend
    fig_happiness = px.line(
        historical_data,
        x='Year',
        y='Ladder score',
        title=f"Happiness Score Trend for {selected_country}",
        markers=True
    )
    fig_happiness.update_traces(line_color="#3498db", line_width=3)
    fig_happiness.update_layout(
        height=400,
        xaxis_title="Year",
        yaxis_title="Happiness Score",
        hovermode='x unified'
    )
    st.plotly_chart(fig_happiness, use_container_width=True)
    
    # Create multi-line chart for all factors
    fig_factors = go.Figure()
    
    for metric in [m for m in selected_metrics if m != 'Ladder score']:
        # Normalize values for better comparison
        values = historical_data[metric]
        normalized = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else values
        
        fig_factors.add_trace(go.Scatter(
            x=historical_data['Year'],
            y=normalized,
            name=metric,
            mode='lines+markers'
        ))
    
    fig_factors.update_layout(
        title=f"Normalized Factor Trends for {selected_country}",
        xaxis_title="Year",
        yaxis_title="Normalized Value",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig_factors, use_container_width=True)
    
    # Year-over-year changes
    if len(historical_data) > 1:
        st.subheader("Year-over-Year Changes")
        
        # Calculate year-over-year changes
        yoy_changes = pd.DataFrame()
        for metric in selected_metrics:
            yoy_changes[metric] = historical_data[metric].pct_change() * 100
        
        # Display recent changes
        recent_changes = yoy_changes.iloc[-1].round(2)
        cols = st.columns(3)
        
        for idx, (metric, change) in enumerate(recent_changes.items()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="profile-card">
                    <div class="metric-label">{metric}</div>
                    <div class="metric-value">
                        <span class="{'trend-up' if change >= 0 else 'trend-down'}">
                            {change:+.2f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.subheader("Factor Analysis")
    
    # Current year's factor breakdown
    fig_breakdown = go.Figure()
    
    # Prepare data for factor breakdown
    factors = [m for m in selected_metrics if m != 'Ladder score']
    values = [current_data[m] for m in factors]
    
    # Create horizontal bar chart
    fig_breakdown.add_trace(go.Bar(
        y=factors,
        x=values,
        orientation='h',
        marker_color='#3498db',
        text=[f'{v:.3f}' for v in values],
        textposition='auto',
    ))
    
    fig_breakdown.update_layout(
        title=f"Factor Breakdown for {selected_country} ({selected_year})",
        xaxis_title="Value",
        yaxis_title="Factors",
        height=400,
        margin=dict(l=200)  # Add more margin for factor names
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Factor contribution analysis
    st.subheader("Factor Impact Analysis")
    
    # Calculate global averages and standard deviations
    global_stats = df[df['Year'] == selected_year][factors].agg(['mean', 'std'])
    
    # Create factor impact cards
    cols = st.columns(2)
    for idx, factor in enumerate(factors):
        with cols[idx % 2]:
            value = current_data[factor]
            avg = global_stats.loc['mean', factor]
            std = global_stats.loc['std', factor]
            z_score = (value - avg) / std if std != 0 else 0
            
            # Determine impact level and colors
            if abs(z_score) > 2:
                impact = "Very High"
                color = "#e74c3c" if z_score < 0 else "#27ae60"
                bg_color = "#fde8e7" if z_score < 0 else "#e8f7f0"
                text_color = "#c0392b" if z_score < 0 else "#218c5e"
            elif abs(z_score) > 1:
                impact = "High"
                color = "#e67e22"
                bg_color = "#fdf3e8"
                text_color = "#d35400"
            else:
                impact = "Moderate"
                color = "#f1c40f"
                bg_color = "#fdf9e8"
                text_color = "#927608"
            
            st.markdown(f"""
            <div class="profile-card">
                <div style="margin-bottom: 1rem;">
                    <h3 style="color: #2c3e50; font-size: 1.2rem; margin: 0; padding: 0;">
                        {factor}
                    </h3>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Current Value</span>
                    <span class="metric-value">{value:.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Global Average</span>
                    <span class="metric-value">{avg:.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Impact Level</span>
                    <span class="metric-value" style="color: {color}; font-weight: bold;">{impact}</span>
                </div>
                <div style="margin-top: 0.5rem; text-align: center; background-color: {bg_color}; 
                          padding: 0.75rem; border-radius: 4px; border: 1px solid {color}40;">
                    <div style="color: {text_color}; font-weight: 500;">
                        {abs(z_score):.1f} standard deviations {('above' if z_score > 0 else 'below')} average
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("Factor Correlations")
    
    # Calculate correlations between factors
    correlations = df[df['Year'] == selected_year][['Ladder score'] + factors].corr()
    
    # Create heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlations.values,
        x=correlations.columns,
        y=correlations.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlations.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_corr.update_layout(
        title="Factor Correlation Matrix",
        height=500,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Add correlation insights
    st.markdown("### Key Insights")
    ladder_corr = correlations['Ladder score'].sort_values(ascending=False)[1:4]  # Top 3 correlations
    st.markdown("Top factors correlated with happiness score:")
    for factor, corr in ladder_corr.items():
        st.markdown(f"- **{factor}**: {corr:.3f} correlation coefficient")

with tab4:
    st.subheader("Peer Comparison")
    
    # Find similar countries based on selected metrics
    def calculate_similarity(row):
        return np.sqrt(np.mean([(row[m] - current_data[m])**2 for m in selected_metrics]))
    
    # Get data for the current year
    year_data = df[df['Year'] == selected_year].copy()
    
    # Calculate similarities
    year_data['similarity'] = year_data.apply(calculate_similarity, axis=1)
    
    # Get top 5 most similar countries (excluding the selected country)
    similar_countries = year_data[year_data['Country name'] != selected_country].nsmallest(5, 'similarity')
    
    # Display similar countries
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown("### Most Similar Countries")
    st.markdown("Based on selected metrics, these countries have the most similar profiles:")
    
    for _, country in similar_countries.iterrows():
        similarity_pct = (1 - country['similarity']) * 100
        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-label">{country['Country name']}</span>
            <span class="metric-value">{similarity_pct:.1f}% similar</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparative Analysis
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown("### Comparative Analysis")
    
    # Calculate z-scores for the selected country
    z_scores = {}
    for metric in selected_metrics:
        mean = year_data[metric].mean()
        std = year_data[metric].std()
        z_scores[metric] = (current_data[metric] - mean) / std if std != 0 else 0
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add bars for z-scores
    fig.add_trace(go.Bar(
        x=list(z_scores.keys()),
        y=list(z_scores.values()),
        marker_color=['#27ae60' if z >= 0 else '#e74c3c' for z in z_scores.values()],
        text=[f"{z:.2f}σ" for z in z_scores.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Performance Relative to Global Average",
        yaxis_title="Standard Deviations from Mean",
        showlegend=False,
        height=400,
        margin=dict(t=30, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    for metric, z_score in z_scores.items():
        if abs(z_score) > 2:
            performance = "significantly above" if z_score > 0 else "significantly below"
            st.markdown(f"- {metric} is {performance} the global average")
    
    st.markdown("</div>", unsafe_allow_html=True) 