import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import traceback
from typing import Optional, Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Predictive Analytics", page_icon="🔮", layout="wide")

# Custom CSS for better error messages and UI consistency with other pages
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
    .success-message {
        color: #27ae60;
        padding: 1rem;
        background-color: #d4efdf;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        color: #f39c12;
        padding: 1rem;
        background-color: #fef9e7;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Display error function for consistent error handling
def display_error(message: str) -> None:
    """Display formatted error message"""
    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

# Display info function for consistent info messaging
def display_info(message: str) -> None:
    """Display formatted info message"""
    st.markdown(f'<div class="info-message">{message}</div>', unsafe_allow_html=True)

# Display success function for consistent success messaging
def display_success(message: str) -> None:
    """Display formatted success message"""
    st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)

# Display warning function for consistent warning messaging
def display_warning(message: str) -> None:
    """Display formatted warning message"""
    st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)

# Load data with better error handling
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Load and validate the happiness data CSV file.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame containing happiness data or None if error occurs
    """
    try:
        if not os.path.exists("happy.csv"):
            display_error("Data file 'happy.csv' not found. Please ensure the file exists in the application directory.")
            return None
        
        df = pd.read_csv("happy.csv")
        
        # Validate required columns exist
        features = ['gdp', 'social_support', 'life_expectancy', 
                   'freedom_to_make_life_choices', 'generosity', 'corruption']
        required_columns = ['country', 'happiness'] + features
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            display_error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
            return None
            
        # Convert numeric columns to appropriate types
        numeric_cols = ['happiness'] + features
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and handle missing values
        missing_count = df[numeric_cols].isnull().sum().sum()
        if missing_count > 0:
            display_warning(f"Dataset contains {missing_count} missing values across {len(numeric_cols)} columns. They will be filled with median values.")
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        display_error(f"Error loading data: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Train model with better error handling and more options
@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series, model_type: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Dict[str, float], Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train a prediction model with the specified algorithm.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model to train ('rf', 'linear', 'ridge', 'lasso')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Trained model
        - Dictionary of model metrics
        - Dictionary of feature importances
        - Training and test data splits
        - Test predictions
    """
    try:
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Initialize appropriate model based on type
        if model_type == 'rf':
            # Random Forest with pipeline for preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=random_state))
            ])
            
            # Simple grid search for hyperparameters
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
        elif model_type == 'linear':
            # Linear Regression with pipeline for preprocessing
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            
        elif model_type == 'ridge':
            # Ridge Regression with pipeline for preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=random_state))
            ])
            
            # Simple grid search for hyperparameters
            param_grid = {
                'model__alpha': [0.1, 1.0, 10.0]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
        elif model_type == 'lasso':
            # Lasso Regression with pipeline for preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(random_state=random_state))
            ])
            
            # Simple grid search for hyperparameters
            param_grid = {
                'model__alpha': [0.1, 1.0, 10.0]
            }
            
            model = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # If it's a grid search model, get the best model
        if hasattr(model, 'best_estimator_'):
            best_model = model.best_estimator_
        else:
            best_model = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae
        }
        
        # Feature importances
        feature_importances = {}
        
        if model_type == 'rf':
            if hasattr(model, 'best_estimator_'):
                # Extract feature importances from best RandomForest model
                importances = model.best_estimator_.named_steps['model'].feature_importances_
                std = np.std([tree.feature_importances_ for tree in model.best_estimator_.named_steps['model'].estimators_], axis=0)
                feature_importances = {
                    'importances': importances,
                    'std': std
                }
            else:
                importances = model.named_steps['model'].feature_importances_
                std = np.std([tree.feature_importances_ for tree in model.named_steps['model'].estimators_], axis=0)
                feature_importances = {
                    'importances': importances,
                    'std': std
                }
        else:
            # For linear models, compute permutation importance
            try:
                perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state)
                feature_importances = {
                    'importances': perm_importance.importances_mean,
                    'std': perm_importance.importances_std
                }
            except Exception as e:
                st.warning(f"Could not calculate permutation importance: {str(e)}")
                # Use coefficients for linear models as a fallback
                if hasattr(model, 'best_estimator_'):
                    if hasattr(model.best_estimator_.named_steps['model'], 'coef_'):
                        coefs = np.abs(model.best_estimator_.named_steps['model'].coef_)
                        feature_importances = {
                            'importances': coefs,
                            'std': np.zeros_like(coefs)
                        }
                elif hasattr(model, 'named_steps'):
                    if hasattr(model.named_steps['model'], 'coef_'):
                        coefs = np.abs(model.named_steps['model'].coef_)
                        feature_importances = {
                            'importances': coefs,
                            'std': np.zeros_like(coefs)
                        }
        
        display_success(f"Model trained successfully with {model_type.upper()} algorithm.")
        return model, metrics, feature_importances, X_train, X_test, y_train, y_test, y_pred
        
    except Exception as e:
        display_error(f"Error training model: {str(e)}")
        st.error(traceback.format_exc())
        raise

def predict_happiness(model, input_data: Dict[str, float], feature_names: List[str]) -> float:
    """
    Make happiness prediction using the trained model.
    
    Args:
        model: Trained model
        input_data: Dictionary of feature values
        feature_names: List of feature names
        
    Returns:
        float: Predicted happiness score
    """
    try:
        # Convert input dictionary to DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return prediction
    except Exception as e:
        display_error(f"Error making prediction: {str(e)}")
        st.error(traceback.format_exc())
        return None

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, std: np.ndarray) -> go.Figure:
    """
    Create a feature importance plot.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        std: Array of standard deviations for feature importances
        
    Returns:
        go.Figure: Plotly figure for feature importance
    """
    try:
        # Create indices for features
        indices = np.argsort(importances)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for feature importances
        fig.add_trace(go.Bar(
            y=[feature_names[i] for i in indices],
            x=importances[indices],
            orientation='h',
            error_x=dict(
                type='data',
                array=std[indices],
                visible=True
            ),
            marker_color='rgb(55, 83, 109)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        display_error(f"Error creating feature importance plot: {str(e)}")
        return None

def plot_predictions(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        
    Returns:
        go.Figure: Plotly figure for actual vs predicted scatter plot
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(
                color='rgb(55, 83, 109)',
                size=8
            ),
            name='Predictions'
        ))
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(
                color='red',
                dash='dash'
            ),
            name='Perfect Prediction'
        ))
        
        # Update layout
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Happiness',
            yaxis_title='Predicted Happiness',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        display_error(f"Error creating prediction plot: {str(e)}")
        return None

def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """
    Create a residual plot.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        
    Returns:
        go.Figure: Plotly figure for residual plot
    """
    try:
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                color='rgb(55, 83, 109)',
                size=8
            )
        ))
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=min(y_pred),
            y0=0,
            x1=max(y_pred),
            y1=0,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        display_error(f"Error creating residual plot: {str(e)}")
        return None

def display_metrics(metrics: Dict[str, float]) -> None:
    """
    Display model metrics in a formatted way.
    
    Args:
        metrics: Dictionary of model metrics
    """
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
        
        with col2:
            st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
        
        with col3:
            st.metric("R² Score", f"{metrics['R2']:.4f}")
        
        with col4:
            st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
            
        # Add explanations of metrics
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            - **Mean Squared Error (MSE)**: Average of the squared differences between predictions and actual values. Lower is better.
            - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as the target variable. Lower is better.
            - **R² Score**: Proportion of variance in the dependent variable explained by the model (1.0 is perfect prediction). Higher is better.
            - **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values. Lower is better.
            """)
            
    except Exception as e:
        display_error(f"Error displaying metrics: {str(e)}")

def main():
    """Main function for the Predictive Analytics page."""
    
    # Page title and description
    st.title("🔮 Predictive Analytics")
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <h3 style='color: #2c3e50;'>Predict Happiness Based on Country Factors</h3>
        <p>Train machine learning models to understand what drives happiness and make predictions for different scenarios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Define features and target
    features = ['gdp', 'social_support', 'life_expectancy', 
               'freedom_to_make_life_choices', 'generosity', 'corruption']
    target = 'happiness'
    
    # Create sidebar for model selection and training
    with st.sidebar:
        st.header("🧠 Model Training")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "Linear Regression", "Ridge Regression", "Lasso Regression"],
            index=0,
            help="Choose which type of model to train"
        )
        
        # Map friendly names to model codes
        model_code_map = {
            "Random Forest": "rf",
            "Linear Regression": "linear",
            "Ridge Regression": "ridge",
            "Lasso Regression": "lasso"
        }
        
        model_code = model_code_map[model_type]
        
        # Feature selection
        st.subheader("Feature Selection")
        selected_features = []
        
        # Create columns for better layout of checkboxes
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(features):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                if st.checkbox(feature.replace('_', ' ').title(), value=True, key=f"feature_{feature}"):
                    selected_features.append(feature)
        
        if not selected_features:
            st.warning("Please select at least one feature")
            return
            
        # Test size slider
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        # Random state for reproducibility
        random_state = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=1000,
            value=42,
            help="Random seed for reproducibility"
        )
        
        # Train button
        train_button = st.button("Train Model")

    # Main content area
    with st.container():
        # Train model when button is clicked
        if train_button:
            with st.spinner(f"Training {model_type}..."):
                try:
                    # Get feature matrix and target
                    X = df[selected_features].copy()
                    y = df[target].copy()
                    
                    # Train model
                    model_results = train_model(X, y, model_code, test_size, int(random_state))
                    
                    if model_results:
                        model, metrics, feature_importances, X_train, X_test, y_train, y_test, y_pred = model_results
                        
                        # Store model and feature names in session state for prediction
                        st.session_state.model = model
                        st.session_state.feature_names = selected_features
                        st.session_state.metrics = metrics
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred
                        st.session_state.feature_importances = feature_importances
                        
                        # Display success message
                        display_success(f"{model_type} trained successfully!")
                    else:
                        display_error("Model training failed.")
                        
                except Exception as e:
                    display_error(f"Error training model: {str(e)}")
                    st.error(traceback.format_exc())
    
        # Display model results if a model has been trained
        if hasattr(st.session_state, 'model'):
            st.header("Model Evaluation")
            
            # Display metrics
            display_metrics(st.session_state.metrics)
            
            # Display visualizations in tabs
            tab1, tab2, tab3 = st.tabs(["Feature Importance", "Predictions", "Residuals"])
            
            with tab1:
                # Display feature importance plot
                try:
                    if 'feature_importances' in st.session_state:
                        fi = st.session_state.feature_importances
                        fig_importance = plot_feature_importance(
                            st.session_state.feature_names,
                            fi['importances'],
                            fi['std']
                        )
                        
                        if fig_importance:
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            with st.expander("What does feature importance mean?"):
                                st.markdown("""
                                Feature importance indicates how much each feature contributes to the model's predictions:
                                
                                - For **Random Forest**, it shows the average reduction in impurity (uncertainty) achieved by splitting on each feature.
                                - For linear models, it shows either the magnitude of coefficients or permutation importance (how much performance drops when a feature is randomly shuffled).
                                
                                Higher values indicate more important features for prediction.
                                """)
                except Exception as e:
                    display_error(f"Error displaying feature importance: {str(e)}")
            
            with tab2:
                # Display prediction plot
                try:
                    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
                        fig_pred = plot_predictions(st.session_state.y_test, st.session_state.y_pred)
                        
                        if fig_pred:
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            with st.expander("How to interpret this plot?"):
                                st.markdown("""
                                The **Actual vs Predicted Values** plot shows how well the model's predictions match the actual values:
                                
                                - Points on the red dashed line represent perfect predictions.
                                - Points above the line indicate the model is overestimating.
                                - Points below the line indicate the model is underestimating.
                                - Ideally, points should be tightly clustered around the line.
                                """)
                except Exception as e:
                    display_error(f"Error displaying prediction plot: {str(e)}")
            
            with tab3:
                # Display residual plot
                try:
                    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
                        fig_resid = plot_residuals(st.session_state.y_test, st.session_state.y_pred)
                        
                        if fig_resid:
                            st.plotly_chart(fig_resid, use_container_width=True)
                            
                            with st.expander("How to interpret residuals?"):
                                st.markdown("""
                                The **Residual Plot** shows the difference between actual and predicted values:
                                
                                - Residuals should be randomly distributed around the zero line.
                                - Patterns in residuals suggest the model isn't capturing some aspect of the data.
                                - Points far from zero are potential outliers or difficult cases for the model.
                                """)
                except Exception as e:
                    display_error(f"Error displaying residual plot: {str(e)}")
            
            # Add prediction section
            st.header("🔮 Make Predictions")
            st.markdown("Adjust the factors below to predict happiness scores for different scenarios.")
            
            try:
                # Create input fields for features
                input_data = {}
                
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                
                feature_defaults = {
                    'gdp': df['gdp'].median(),
                    'social_support': df['social_support'].median(),
                    'life_expectancy': df['life_expectancy'].median(),
                    'freedom_to_make_life_choices': df['freedom_to_make_life_choices'].median(),
                    'generosity': df['generosity'].median(),
                    'corruption': df['corruption'].median()
                }
                
                feature_mins = {f: df[f].min() for f in st.session_state.feature_names}
                feature_maxs = {f: df[f].max() for f in st.session_state.feature_names}
                
                # Create sliders for each feature in the model
                for i, feature in enumerate(st.session_state.feature_names):
                    # Alternate between columns
                    with col1 if i % 2 == 0 else col2:
                        input_data[feature] = st.slider(
                            feature.replace('_', ' ').title(),
                            min_value=float(feature_mins[feature]),
                            max_value=float(feature_maxs[feature]),
                            value=float(feature_defaults[feature]),
                            step=0.01
                        )
                
                # Predict button
                if st.button("Predict Happiness"):
                    with st.spinner("Making prediction..."):
                        prediction = predict_happiness(
                            st.session_state.model,
                            input_data,
                            st.session_state.feature_names
                        )
                        
                        if prediction is not None:
                            # Create a scale from min to max happiness for context
                            min_happiness = df['happiness'].min()
                            max_happiness = df['happiness'].max()
                            
                            # Display the prediction with a gauge
                            st.subheader("Predicted Happiness Score")
                            
                            # Create three columns for layout
                            col1, col2, col3 = st.columns([1, 3, 1])
                            
                            with col2:
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=prediction,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [min_happiness, max_happiness]},
                                        'bar': {'color': "#2c3e50"},
                                        'steps': [
                                            {'range': [min_happiness, min_happiness + (max_happiness - min_happiness)/3], 'color': "#fadbd8"},
                                            {'range': [min_happiness + (max_happiness - min_happiness)/3, 
                                                      min_happiness + 2*(max_happiness - min_happiness)/3], 'color': "#d6eaf8"},
                                            {'range': [min_happiness + 2*(max_happiness - min_happiness)/3, 
                                                      max_happiness], 'color': "#d4efdf"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': prediction
                                        }
                                    }
                                ))
                                
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=20, r=20, t=30, b=20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Find similar countries
                            st.subheader("Similar Countries")
                            st.markdown("Countries with similar predicted happiness scores:")
                            
                            # Find countries with similar happiness scores
                            similar_countries = df.copy()
                            similar_countries['score_diff'] = abs(similar_countries['happiness'] - prediction)
                            similar_countries = similar_countries.sort_values('score_diff').head(5)
                            
                            # Display similar countries in a table
                            st.dataframe(
                                similar_countries[['country', 'happiness']].rename(
                                    columns={'country': 'Country', 'happiness': 'Happiness Score'}
                                ),
                                use_container_width=True
                            )
                            
                            # Add explanation of the prediction
                            with st.expander("What does this prediction mean?"):
                                st.markdown(f"""
                                The predicted happiness score of **{prediction:.2f}** suggests that, based on the selected factors, 
                                the country would have a happiness level similar to the following countries:
                                
                                - {', '.join(similar_countries['country'].values)}
                                
                                This score is based on the model's understanding of how these factors contribute to happiness.
                                """)
            except Exception as e:
                display_error(f"Error making prediction: {str(e)}")
                st.error(traceback.format_exc())
                return  
    # Footer
    st.markdown("""
    <div style='background-color: #000000; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>  
        <p style='text-align: center; color: #2c3e50#f8f9fa;'>© 2023 Predictive Analytics App</p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()