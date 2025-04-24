import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page config
st.set_page_config(page_title="Happiness Clusters", page_icon="🔍", layout="wide")

# Title and documentation
st.title("🔍 Happiness Cluster Analysis")

with st.expander("📚 About Cluster Analysis"):
    st.markdown("""
    ### Understanding Happiness Clusters
    
    This analysis uses machine learning (K-means clustering) to identify groups of countries with similar happiness characteristics, helping uncover patterns and relationships in global happiness data.
    
    #### Key Features:
    1. **Dynamic Clustering**
       - Adjustable number of clusters (2-8)
       - Flexible feature selection
       - Real-time visualization updates
    
    2. **Interactive Visualizations**
       - PCA-based 2D cluster plot
       - Radar charts for cluster profiles
       - Heatmap of cluster characteristics
    
    3. **Detailed Analysis**
       - Cluster statistics and summaries
       - Key distinguishing features
       - Country grouping patterns
    
    4. **Statistical Insights**
       - Cluster stability metrics
       - Feature importance analysis
       - Inter-cluster relationships
    """)

with st.expander("🔍 How to Use This Tool"):
    st.markdown("""
    ### Guide to Cluster Analysis
    
    #### Step-by-Step Instructions:
    1. **Configure Parameters**
       - Select analysis year
       - Choose number of clusters (2-8)
       - Pick features for clustering
    
    2. **Interpret Visualizations**
       - PCA Plot: Shows country groupings
       - Radar Chart: Displays cluster characteristics
       - Heatmap: Reveals feature patterns
    
    3. **Analyze Results**
       - Review cluster profiles
       - Examine country groupings
       - Identify distinguishing features
    
    #### Analysis Tips:
    - Start with fewer clusters (3-4) for clear patterns
    - Include diverse features for robust clustering
    - Look for geographical or economic patterns
    - Consider cultural and regional contexts
    """)

with st.expander("📊 Understanding the Methods"):
    st.markdown("""
    ### Technical Methodology
    
    #### Clustering Algorithm
    - **K-means Clustering**
      - Iterative centroid-based clustering
      - Minimizes within-cluster variance
      - Optimal for discovering natural groupings
    
    #### Data Processing
    1. **Feature Scaling**
       - StandardScaler for normalization
       - Mean = 0, Standard deviation = 1
       - Ensures fair feature comparison
    
    2. **Dimensionality Reduction**
       - PCA for visualization
       - 2D projection of high-dimensional data
       - Preserves major variance patterns
    
    #### Interpretation Guide:
    - **Cluster Size**: Number of countries in each group
    - **Centroids**: Average characteristics of clusters
    - **Distance**: Similarity between countries/clusters
    - **Feature Importance**: Key distinguishing factors
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
    }
    .cluster-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    .cluster-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    .cluster-card p {
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    .cluster-card ul {
        list-style-type: none;
        padding-left: 0;
        margin-top: 0.5rem;
    }
    .cluster-card li {
        color: #34495e;
        padding: 0.2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("happiness_by_country_cleaned.csv")
    df.columns = [col.replace('Explained by: ', '') for col in df.columns]
    return df

# Main content
st.markdown("""
Discover patterns and groups of countries with similar happiness characteristics using machine learning clustering techniques.
""")

# Load data
df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Clustering Parameters")
    
    # Year selection
    selected_year = st.selectbox(
        "Select Year",
        options=sorted(df['Year'].unique(), reverse=True),
        help="Choose the year for cluster analysis"
    )
    
    # Number of clusters
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=8,
        value=4,
        help="Select the number of country groups to identify"
    )
    
    # Feature selection
    features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy',
               'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    
    selected_features = st.multiselect(
        "Select Features for Clustering",
        options=features,
        default=features,
        help="Choose which metrics to use for grouping countries"
    )

# Main analysis
if selected_features:
    # Filter data for selected year
    year_data = df[df['Year'] == selected_year].copy()
    
    # Prepare data for clustering
    X = year_data[selected_features]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to data
    year_data['Cluster'] = clusters
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA visualization
    st.subheader("Country Clusters Visualization")
    
    # Create scatter plot
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=clusters,
        hover_name=year_data['Country name'],
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
        title=f"Country Clusters ({selected_year})"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    # Calculate cluster means
    cluster_means = year_data.groupby('Cluster')[selected_features + ['Ladder score']].mean()
    
    # Create radar chart for cluster profiles
    fig_radar = go.Figure()
    
    for cluster in range(n_clusters):
        values = cluster_means.loc[cluster].values.tolist()
        # Add first value again to close the polygon
        values = values + [values[0]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=selected_features + ['Ladder score'] + [selected_features[0]],
            name=f'Cluster {cluster}',
            fill='toself'
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Cluster Profiles"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Display cluster details
    cols = st.columns(n_clusters)
    for i in range(n_clusters):
        with cols[i]:
            cluster_countries = year_data[year_data['Cluster'] == i]['Country name'].tolist()
            avg_happiness = year_data[year_data['Cluster'] == i]['Ladder score'].mean()
            
            st.markdown(f"""
            <div class="cluster-card">
                <h4>Cluster {i + 1}</h4>
                <p><strong>Average Happiness:</strong> {avg_happiness:.2f}</p>
                <p><strong>Top Countries:</strong></p>
                <ul>
                    {"".join([f"<li>• {country}</li>" for country in sorted(cluster_countries)[:5]])}
                </ul>
                <p><strong>Total Countries:</strong> {len(cluster_countries)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Cluster stability analysis
    st.subheader("Cluster Stability Analysis")
    
    # Calculate cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=selected_features
    )
    
    # Create heatmap of cluster centers
    fig_heatmap = px.imshow(
        cluster_centers,
        labels=dict(x="Features", y="Cluster", color="Value"),
        title="Cluster Centers Heatmap"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Display cluster summary statistics
    st.subheader("Cluster Summary Statistics")
    
    summary_stats = year_data.groupby('Cluster')[selected_features + ['Ladder score']].agg([
        'mean', 'std', 'min', 'max'
    ]).round(3)
    
    st.dataframe(summary_stats)
    
    # Cluster interpretation
    st.subheader("Cluster Interpretation")
    
    for i in range(n_clusters):
        cluster_data = year_data[year_data['Cluster'] == i]
        
        # Find distinguishing features
        cluster_profile = cluster_means.loc[i]
        overall_means = year_data[selected_features + ['Ladder score']].mean()
        differences = (cluster_profile - overall_means) / overall_means
        
        # Sort features by absolute difference
        key_features = differences.abs().sort_values(ascending=False)[:3]
        
        st.markdown(f"""
        #### Cluster {i +1} Characteristics:
        - **Size**: {len(cluster_data)} countries
        - **Average Happiness**: {cluster_data['Ladder score'].mean():.2f}
        - **Key Features**:
            - {key_features.index[0]}: {differences[key_features.index[0]]:.1%} different from average
            - {key_features.index[1]}: {differences[key_features.index[1]]:.1%} different from average
            - {key_features.index[2]}: {differences[key_features.index[2]]:.1%} different from average
        """)

else:
    st.warning("Please select at least one feature for clustering analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Data source: World Happiness Report</p>
    <p>Cluster analysis performed using K-means clustering algorithm</p>
</div>
""", unsafe_allow_html=True) 