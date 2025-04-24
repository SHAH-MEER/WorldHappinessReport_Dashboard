import pandas as pd
import numpy as np
from datetime import datetime
import os

# Input and output paths
RAW_DATA_PATH = "data/raw/happiness_by_country.csv"
PROCESSED_DATA_PATH = "data/processed/happiness_by_country_cleaned.csv"
LATEST_DATA_PATH = "data/processed/latest_happiness_cleaned.csv"
OUTPUT_PATH = "data/processed/output.csv"
SUMMARY_PATH = "data/processed/happiness_summary_detailed.txt"

def clean_column_names(df):
    # First, create a copy of the dataframe
    df = df.copy()
    
    # Rename the columns using a direct mapping
    name_mapping = {
        'Country name': 'country_name',
        'Ladder score': 'ladder_score',
        'Explained by: Log GDP per capita': 'log_gdp_per_capita',
        'Explained by: Social support': 'social_support',
        'Explained by: Healthy life expectancy': 'healthy_life_expectancy',
        'Explained by: Freedom to make life choices': 'freedom_to_make_life_choices',
        'Explained by: Generosity': 'generosity',
        'Explained by: Perceptions of corruption': 'perceptions_of_corruption',
        'Dystopia + residual': 'dystopia_residual',
        'Year': 'year',
        'Rank': 'rank',
        'upperwhisker': 'upper_whisker',
        'lowerwhisker': 'lower_whisker'
    }
    
    # Rename columns using the mapping
    df = df.rename(columns=name_mapping)
    
    return df

def handle_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clean

def calculate_happiness_changes(df):
    # Sort by country and year
    df = df.sort_values(['country_name', 'year'])
    
    # Calculate year-over-year changes
    df['happiness_change'] = df.groupby('country_name')['ladder_score'].diff()
    
    # Calculate 5-year changes for countries
    df['happiness_change_5yr'] = df.groupby('country_name')['ladder_score'].diff(5)
    
    return df

def process_happiness_data(file_path):
    # Read the data
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")
    
    # Convert to string type first
    df['Country name'] = df['Country name'].astype(str)
    
    # Sort by country and year
    df = df.sort_values(['Country name', 'Year'])
    
    # Fill missing values within each country's timeline
    for col in df.columns:
        if col not in ['Year', 'Rank', 'Country name']:
            # First forward fill within groups
            df[col] = df.groupby('Country name')[col].transform(lambda x: x.ffill())
            # Then backward fill within groups
            df[col] = df.groupby('Country name')[col].transform(lambda x: x.bfill())
    
    # Remove countries with less than 5 years of data
    country_counts = df.groupby('Country name').size()
    valid_countries = country_counts[country_counts >= 5].index
    df = df[df['Country name'].isin(valid_countries)]
    
    # Handle outliers in numeric columns
    numeric_cols = ['Ladder score', 'Explained by: Log GDP per capita', 
                   'Explained by: Social support', 'Explained by: Healthy life expectancy']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Calculate happiness changes
    df['Happiness Change'] = df.groupby('Country name')['Ladder score'].diff()
    df['5-Year Change'] = df.groupby('Country name')['Ladder score'].diff(5)
    
    # Generate summary statistics
    summary = []
    summary.append("Dataset Summary")
    summary.append("==============\n")
    summary.append(f"Total countries: {df['Country name'].nunique()}")
    summary.append(f"Year range: {df['Year'].min()}-{df['Year'].max()}")
    summary.append(f"Average happiness score: {df['Ladder score'].mean():.4f}")
    summary.append(f"Maximum happiness score: {df['Ladder score'].max():.3f}")
    summary.append(f"Minimum happiness score: {df['Ladder score'].min():.3f}")
    
    # Data quality report
    summary.append("\nData Quality Metrics")
    summary.append("===================\n")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    summary.append("Missing values per column:")
    for col, count in missing_values.items():
        summary.append(f"{col}: {count} missing values")
    
    # Recent happiness changes
    summary.append("\nCountries with most improved happiness (last 5 years):")
    latest_year = df['Year'].max()
    five_year_changes = df[df['Year'] == latest_year]['5-Year Change']
    top_improved = five_year_changes.nlargest(5)
    for idx, change in zip(top_improved.index, top_improved.values):
        if pd.notna(change):
            country_name = df.loc[idx, 'Country name']
            summary.append(f"{country_name}: {change:.3f}")
    
    # Save files
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    # Save latest data for each country
    latest_data = df.sort_values('Year').groupby('Country name').last()
    latest_data.to_csv(LATEST_DATA_PATH, index=False)
    
    # Save summary
    with open(SUMMARY_PATH, 'w') as f:
        f.write('\n'.join(summary))
    
    return df

def main():
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Load and process data
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Process data...
    
    # Save processed data
    df_cleaned = process_happiness_data(RAW_DATA_PATH)
    latest_df = df_cleaned.sort_values('Year').groupby('Country name').last()
    output_df = df_cleaned.copy()
    
    # Save processed data
    df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    latest_df.to_csv(LATEST_DATA_PATH, index=False)
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    # Generate and save summary
    summary = generate_summary(df_cleaned)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
    print("Data processing complete!") 