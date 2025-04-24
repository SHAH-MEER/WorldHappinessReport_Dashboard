# World Happiness Dashboard 🌍

An interactive web application built with Streamlit that visualizes and analyzes global happiness data across different countries and regions. The dashboard provides comprehensive analysis tools, interactive visualizations, and detailed insights into factors affecting global happiness.

## Features 🌟

### 1. Country Analysis 🌎
- Detailed country profiles
- Historical trends analysis
- Factor breakdown and impact analysis
- Comparative metrics and rankings

### 2. Regional Analysis 🗺️
- Region-wise happiness comparisons
- Geographic distribution patterns
- Regional trends and patterns
- Interactive map visualizations

### 3. Predictive Analytics 🔮
- Future happiness score predictions
- Factor impact forecasting
- Trend analysis and projections
- Machine learning insights

### 4. Time Series Analysis 📈
- Historical trend visualization
- Seasonal patterns analysis
- Year-over-year comparisons
- Growth rate analysis

### 5. Country Comparison 🔄
- Side-by-side country comparison
- Factor-wise comparative analysis
- Radar charts for visual comparison
- Similarity scoring

### 6. Cluster Analysis 🔍
- Country clustering based on factors
- Cluster characteristics analysis
- Interactive cluster visualization
- Similarity patterns

### 7. Correlation Analysis 📊
- Factor correlation heatmaps
- Detailed correlation analysis
- Network visualization of relationships
- Statistical significance testing

## Project Structure 📁

```
WorldHappinessReport/
├── app.py                      # Main Streamlit application
├── pages/                      # Streamlit pages
│   ├── 1_🌎_Country_Analysis.py
│   ├── 2_🌎_Regional_Analysis.py
│   ├── 3_🔮_Predictive_Analytics.py
│   ├── 4_📈_Time_Series_Analysis.py
│   ├── 5_🔄_Country_Comparison.py
│   ├── 6_🔍_Cluster_Analysis.py
│   └── 7_📊_Correlation_Analysis.py
├── data/                       # Data directory
│   ├── raw/                    # Raw data files
│   │   ├── happiness_by_country.csv
│   │   └── happy.csv
│   └── processed/              # Processed data files
│       ├── happiness_by_country_cleaned.csv
│       ├── latest_happiness_cleaned.csv
│       ├── output.csv
│       └── happiness_summary_detailed.txt
├── process_data.py            # Data processing script
├── requirements.txt           # Project dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/SHAH-MEER/WorldHappinessReport.git
cd WorldHappinessReport
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Process the data (if needed):
```bash
python process_data.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the dashboard at:
```
http://localhost:8501
```

## Data Sources 📊

The application uses data from the World Happiness Report, which includes:
- Happiness scores by country
- GDP per capita
- Social support metrics
- Healthy life expectancy
- Freedom to make life choices
- Generosity
- Perceptions of corruption

## Dependencies 📦

Main dependencies include:
- streamlit>=1.24.0
- plotly>=5.13.1
- pandas>=1.5.3
- numpy>=1.24.3
- scikit-learn>=1.2.2
- seaborn>=0.12.2

For a complete list, see `requirements.txt`.

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 👏

- World Happiness Report for the comprehensive dataset
- Streamlit team for the excellent web framework
- Plotly team for interactive visualizations
- The open-source community for various tools and libraries

## Contact 📧

Shahmeer - [@shahmeer](https://github.com/SHAH-MEER)

Project Link: [https://github.com/SHAH-MEER/WorldHappinessReport](https://github.com/SHAH-MEER/WorldHappinessReport)

---
