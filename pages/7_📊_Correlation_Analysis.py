import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import seaborn as sns
import pingouin as pg  # for partial correlations
import networkx as nx  # for correlation network

# Page config
st.set_page_config(page_title="Correlation Analysis", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    :root {
        --bg-color: #ffffff;
        --text-color: #2c3e50;
        --card-bg: #ffffff;
        --card-border: #e0e0e0;
        --hover-shadow: rgba(0,0,0,0.1);
        --secondary-bg: #f8f9fa;
        --strong-color: #27ae60;
        --moderate-color: #f39c12;
        --weak-color: #e74c3c;
        --strong-bg: #e8f5e9;
        --moderate-bg: #fef6e7;
        --weak-bg: #fde8e7;
    }
    
    .correlation-card {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid var(--card-border);
        box-shadow: 0 4px 6px var(--hover-shadow);
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.75rem 0;
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-bg);
    }
    
    .metric-label {
        color: var(--text-color);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .metric-value {
        font-weight: 600;
        color: #2c3e50;
    }
    
    .insight-box {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--card-border);
    }
    
    .correlation-strong {
        color: var(--strong-color);
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.25rem 0.75rem;
        background-color: var(--strong-bg);
        border-radius: 0.25rem;
    }
    
    .correlation-moderate {
        color: var(--moderate-color);
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.25rem 0.75rem;
        background-color: var(--moderate-bg);
        border-radius: 0.25rem;
    }
    
    .correlation-weak {
        color: var(--weak-color);
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.25rem 0.75rem;
        background-color: var(--weak-bg);
        border-radius: 0.25rem;
    }
    
    .p-value-text {
        color: var(--text-color);
        font-size: 0.9rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_partial_correlations(data, variables, covariate):
    """Calculate partial correlations controlling for a covariate"""
    partial_corrs = {}
    for var1 in variables:
        for var2 in variables:
            if var1 != var2:
                result = pg.partial_corr(data=data, x=var1, y=var2, covar=covariate)
                partial_corrs[(var1, var2)] = {
                    'r': result['r'].iloc[0],
                    'p-val': result['p-val'].iloc[0]
                }
    return partial_corrs

def create_correlation_network(corr_matrix, threshold=0.3):
    """Create a correlation network graph"""
    G = nx.Graph()
    
    # Add nodes
    for col in corr_matrix.columns:
        G.add_node(col)
    
    # Add edges for correlations above threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    weight=abs(corr),
                    color='blue' if corr > 0 else 'red'
                )
    
    return G

# Main content
st.title("📊 Correlation Analysis")

# Add comprehensive documentation
with st.expander("📚 About This Analysis"):
    st.markdown("""
    ### Understanding Correlation Analysis
    
    This page provides a comprehensive analysis of relationships between different happiness factors. The analysis helps understand how various factors influence and interact with each other in contributing to overall happiness scores.
    
    #### Key Features:
    - Interactive correlation heatmap
    - Detailed factor-by-factor analysis
    - Network visualization of relationships
    - Statistical significance testing
    - Multiple correlation methods
    
    #### Available Tools:
    1. **Correlation Matrix**: Visual representation of all factor relationships
    2. **Detailed Analysis**: Deep dive into specific factor relationships
    3. **Factor Relationships**: Network graph showing significant connections
    
    #### Methodology:
    - **Pearson Correlation**: Measures linear relationships between variables
    - **Spearman Correlation**: Measures monotonic relationships (rank-based)
    - **P-value Analysis**: Tests statistical significance of correlations
    - **Network Analysis**: Visualizes relationships above significance threshold
    
    #### Interpretation Guide:
    - **Strong Correlation**: |r| > 0.7
    - **Moderate Correlation**: 0.4 < |r| < 0.7
    - **Weak Correlation**: |r| < 0.4
    - **P-value**: Lower values indicate higher statistical significance
    """)

st.markdown("""
Explore relationships between different happiness factors and understand their interdependencies.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("happiness_by_country_cleaned.csv")
    df.columns = [col.replace('Explained by: ', '') for col in df.columns]
    return df

df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Analysis Controls")
    
    with st.expander("ℹ️ Control Panel Guide"):
        st.markdown("""
        ### How to Use Controls
        
        #### Year Selection
        - Choose specific year for analysis
        - More recent years have more complete data
        - Compare different years to see trends
        
        #### Correlation Method
        - **Pearson**: Best for linear relationships
        - **Spearman**: Better for non-linear relationships
        - Choose based on your data assumptions
        
        #### Significance Level
        - Standard levels: 0.01, 0.05, 0.10
        - Lower values = more stringent criteria
        - 0.05 is commonly used in research
        """)
    
    # Year selection
    selected_year = st.selectbox(
        "Select Year",
        options=sorted(df['Year'].unique(), reverse=True),
        help="Choose the year for analysis"
    )
    
    # Correlation method
    correlation_method = st.selectbox(
        "Correlation Method",
        options=["Spearman", "Pearson"],
        help="Choose the correlation calculation method. Spearman is more robust to outliers and non-linear relationships."
    )
    
    # Significance level
    significance_level = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Statistical significance threshold"
    )

# Get data for selected year
year_data = df[df['Year'] == selected_year]

# Available metrics
metrics = [
    'Ladder score', 'Log GDP per capita', 'Social support',
    'Healthy life expectancy', 'Freedom to make life choices',
    'Generosity', 'Perceptions of corruption'
]

# Calculate correlations
if correlation_method == "Pearson":
    corr_matrix = year_data[metrics].corr(method='pearson')
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=metrics, index=metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if i != j:
                _, p_value = stats.pearsonr(year_data[metrics[i]], year_data[metrics[j]])
                p_values.iloc[i, j] = p_value
else:
    corr_matrix = year_data[metrics].corr(method='spearman')
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=metrics, index=metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if i != j:
                _, p_value = stats.spearmanr(year_data[metrics[i]], year_data[metrics[j]])
                p_values.iloc[i, j] = p_value

# Create tabs
tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Detailed Analysis", "Factor Relationships"])

with tab1:
    with st.expander("📊 Understanding the Correlation Matrix"):
        st.markdown("""
        ### How to Read the Correlation Matrix
        
        #### Heatmap Interpretation
        - **Colors**: Blue = positive correlation, Red = negative correlation
        - **Intensity**: Darker colors indicate stronger relationships
        - **Values**: Range from -1 (perfect negative) to +1 (perfect positive)
        
        #### Key Features
        1. **Diagonal**: Always 1.0 (perfect correlation with itself)
        2. **Symmetry**: Matrix is symmetrical across diagonal
        3. **Significance**: Marked correlations pass statistical testing
        
        #### Tips for Analysis
        - Look for dark colors to identify strong relationships
        - Check p-values for statistical significance
        - Consider both positive and negative correlations
        - Focus on relationships relevant to your research
        """)
    
    st.subheader("Correlation Heatmap")
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": "#ffffff"},
        hoverongaps=False
    ))
    
    # Update layout for theme
    fig.update_layout(
        title=f"{correlation_method} Correlation Matrix ({selected_year})",
        height=600,
        width=800,
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add significance indicators
    st.markdown("### Significant Correlations")
    significant_corr = []
    
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            corr = corr_matrix.iloc[i, j]
            p_value = p_values.iloc[i, j]
            if p_value < significance_level:
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                significant_corr.append({
                    'pair': f"{metrics[i]} — {metrics[j]}",
                    'correlation': corr,
                    'p_value': p_value,
                    'strength': strength
                })
    
    if significant_corr:
        for corr in sorted(significant_corr, key=lambda x: abs(x['correlation']), reverse=True):
            st.markdown(f"""
            <div class="correlation-card">
                <div class="metric-row">
                    <span class="metric-label">{corr['pair']}</span>
                    <span class="correlation-{corr['strength']}">{corr['correlation']:.3f}</span>
                </div>
                <div class="insight-box">
                    <span class="p-value-text">p-value: {corr['p_value']:.4f} (statistically significant at α = {significance_level})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant correlations found at the selected significance level.")

with tab2:
    with st.expander("🔍 Guide to Detailed Analysis"):
        st.markdown("""
        ### Factor-by-Factor Analysis Guide
        
        #### Scatter Plot Interpretation
        - **Points**: Each represents a country
        - **Trend Line**: Shows overall relationship
        - **Slope**: Indicates relationship strength
        
        #### Analysis Features
        1. **Primary Factor Selection**: Choose main factor to analyze
        2. **Multiple Comparisons**: See relationships with all other factors
        3. **Statistical Summary**: Correlation values and significance
        
        #### How to Use
        1. Select primary factor of interest
        2. Examine scatter plots for patterns
        3. Check correlation summaries
        4. Look for outliers and trends
        """)
    
    st.subheader("Factor-by-Factor Analysis")
    
    # Select primary factor
    primary_factor = st.selectbox(
        "Select Primary Factor",
        options=metrics,
        help="Choose the main factor to analyze"
    )
    
    # Create scatter plots for the selected factor
    correlations = []
    for factor in [m for m in metrics if m != primary_factor]:
        corr = corr_matrix.loc[primary_factor, factor]
        p_value = p_values.loc[primary_factor, factor]
        
        fig = px.scatter(
            year_data,
            x=primary_factor,
            y=factor,
            trendline="ols",
            labels={primary_factor: primary_factor, factor: factor},
            title=f"{primary_factor} vs {factor}"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        correlations.append({
            'factor': factor,
            'correlation': corr,
            'p_value': p_value
        })
    
    # Display correlation summary
    st.markdown("### Correlation Summary")
    for corr in sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True):
        strength = "strong" if abs(corr['correlation']) > 0.7 else "moderate" if abs(corr['correlation']) > 0.4 else "weak"
        st.markdown(f"""
        <div class="correlation-card">
            <div class="metric-row">
                <span class="metric-label">{corr['factor']}</span>
                <span class="correlation-{strength}">{corr['correlation']:.3f}</span>
            </div>
            <div class="insight-box">
                <span class="p-value-text">p-value: {corr['p_value']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    with st.expander("🕸️ Network Visualization Guide"):
        st.markdown("""
        ### Understanding the Network Graph
        
        #### Visual Elements
        - **Nodes**: Represent happiness factors
        - **Edges**: Show significant correlations
        - **Edge Labels**: Display correlation strength
        - **Layout**: Circular for clear visualization
        
        #### Network Properties
        1. **Connection Strength**: Shown by edge labels
        2. **Significance**: Only significant correlations shown
        3. **Centrality**: Factors with many connections are central
        
        #### Analysis Tips
        - Look for clusters of connected factors
        - Identify central/influential factors
        - Consider indirect relationships
        - Focus on strongest connections
        """)
    
    st.subheader("Factor Relationships")
    
    # Create network graph of correlations
    significant_edges = []
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            corr = corr_matrix.iloc[i, j]
            p_value = p_values.iloc[i, j]
            if abs(corr) > 0.4 and p_value < significance_level:  # Show moderate to strong correlations
                significant_edges.append((metrics[i], metrics[j], abs(corr)))
    
    if significant_edges:
        # Create network visualization using plotly
        nodes = list(set([edge[0] for edge in significant_edges] + [edge[1] for edge in significant_edges]))
        edge_x = []
        edge_y = []
        edge_text = []
        
        # Create circular layout with more spacing
        pos = {}
        radius = 1.2  # Increased radius for better spacing
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        
        # Add edges with correlation values
        for edge in significant_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Calculate edge midpoint for text position
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            # Add correlation value text
            edge_text.append(f"r = {edge[2]:.2f}")
        
        # Create the network graph
        fig = go.Figure()
        
        # Add edges with varying width based on correlation strength
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(
                width=2,
                color='rgba(52, 152, 219, 0.5)'  # Light blue with transparency
            ),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add correlation values as text
        for i in range(0, len(edge_x)-2, 3):  # Skip None values
            mid_x = (edge_x[i] + edge_x[i+1]) / 2
            mid_y = (edge_y[i] + edge_y[i+1]) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=edge_text[i//3],
                showarrow=False,
                font=dict(size=10, color='#2c3e50'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                borderpad=2
            )
        
        # Add nodes with improved styling
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=40,
                color='#3498db',
                line=dict(color='#fff', width=2)
            ),
            text=nodes,
            textposition="middle center",
            textfont=dict(size=11, color='white'),
            hoverinfo='text'
        ))
        
        # Update layout for better visualization
        fig.update_layout(
            title=dict(
                text="Factor Relationship Network",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, color='#2c3e50')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=700,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class="correlation-card">
            <p style="color: #2c3e50; font-weight: 600; margin-bottom: 1rem;">How to interpret the network:</p>
            <ul style="color: #2c3e50; margin-left: 1.5rem;">
                <li>Nodes (blue circles) represent happiness factors</li>
                <li>Lines between nodes show significant correlations</li>
                <li>Numbers on lines show correlation strength (r)</li>
                <li>All shown relationships are statistically significant (p < {significance_level})</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No moderate or strong correlations found at the selected significance level.")

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Data source: World Happiness Report</p>
    <p>Analysis includes basic correlations, partial correlations, and network analysis.</p>
    <p>Statistical significance is calculated using p-values (α = 0.05)</p>
</div>
""", unsafe_allow_html=True) 