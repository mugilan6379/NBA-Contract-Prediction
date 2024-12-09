import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Title and Introduction
st.title("NBA Player Contract Prediction App")
st.write("""
This application predicts NBA player contract values based on performance statistics. 
We highlight key steps from data sourcing to modeling and discuss the results and potential uses.
""")

# Data Loading
st.subheader("Data Overview")
st.write("Using pre-loaded datasets for analysis.")

# Load pre-defined datasets
player_data_path = "player_game_data.csv"
team_data_path = "team_game_data.csv"
contract_data_path = "nba_spotrac_data.csv"

# Read the datasets
player_df = pd.read_csv(player_data_path)
team_df = pd.read_csv(team_data_path)
contract_df = pd.read_csv(contract_data_path)

# Display key data highlights
st.write("### Contract Data Sample")
st.write(contract_df.head())
st.write("### Player Game Data Sample")
st.write(player_df.head())
st.write("### Team Game Data Sample")
st.write(team_df.head())

# Data Cleaning and Merging
st.subheader("Data Cleaning and Feature Engineering")
contract_df['cap_hit'] = pd.to_numeric(contract_df['cap_hit'].replace({'\$': '', ',': '', '-': None}, regex=True), errors='coerce')
aggregated_stats = player_df.groupby('player_name').agg({
    'points': ['sum', 'mean'],
    'reboffensive': ['sum', 'mean'],
    'rebdefensive': ['sum', 'mean'],
    'assists': ['sum', 'mean'],
    'steals': ['sum', 'mean'],
    'blocks': ['sum', 'mean']
}).reset_index()
aggregated_stats.columns = ['_'.join(col).strip('_') for col in aggregated_stats.columns]

# Merge data
full_data = pd.merge(contract_df, aggregated_stats, left_on='player', right_on='player_name', how='inner')
full_data.dropna(inplace=True)
st.write("Data successfully cleaned and merged!")

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
st.write("""
### Hypotheses:
1. Players with higher points, assists, and rebounds will have higher contracts.
2. Defensive statistics (steals, blocks) may have a smaller but notable impact.
""")

# Correlation Heatmap
st.write("### Correlation Heatmap")
correlation = full_data[['cap_hit', 'points_sum', 'assists_sum', 'reboffensive_sum', 'rebdefensive_sum', 'steals_sum', 'blocks_sum']].corr()
fig, ax = plt.subplots()
cax = ax.matshow(correlation, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns, rotation=45, ha='left')
ax.set_yticklabels(correlation.columns)
st.pyplot(fig)

# Modeling
st.subheader("Modeling and Evaluation")
st.write("### Model Selection:")
st.write("""
We selected the `HistGradientBoostingRegressor` because:
1. It handles missing data natively.
2. It provides robust predictions with minimal preprocessing.
3. It performs well for structured data.
""")

# Feature and target selection
features = [
    'points_sum', 'points_mean', 'reboffensive_sum', 'rebdefensive_sum',
    'assists_sum', 'steals_sum', 'blocks_sum'
]
target = 'cap_hit'
X = full_data[features]
y = full_data[target]

# Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write("### Results:")
st.write(f"**R² Score:** {r2:.2f} (Explains {r2:.0%} of the variance in contract values)")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f} (Average prediction error)")
st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f} (Typical prediction error magnitude)")

# Visualizations
st.write("### Predictions vs. Actual Contract Values")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual Contract Value')
ax.set_ylabel('Predicted Contract Value')
ax.set_title('Predicted vs. Actual Contract Values')
st.pyplot(fig)

# Error Distribution
st.write("### Error Distribution")
errors = y_test - y_pred
fig, ax = plt.subplots()
ax.hist(errors, bins=30, alpha=0.7, color='blue')
ax.set_xlabel('Prediction Error (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Prediction Errors')
st.pyplot(fig)

# Feature Importance
if hasattr(model, 'feature_importances_'):
    st.write("### Feature Importance")
    importance = model.feature_importances_
    feature_names = features
    fig, ax = plt.subplots()
    ax.barh(feature_names, importance, color='green')
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance in Contract Prediction')
    st.pyplot(fig)

# So What?
st.subheader("So What?")
st.write("""
The model provides insights into which player statistics are most important for determining contract values:
1. **Points and assists** are the strongest predictors, as expected.
2. Defensive metrics like **blocks and steals** have less influence but are still notable.

### Applications:
- Teams could use this model for data-driven contract negotiations.
- Agents might use this to justify higher player valuations.

### Limitations:
- High error values suggest that factors outside performance statistics (e.g., market dynamics, leadership, popularity) significantly influence contracts.
- Future improvements could include incorporating more advanced features like injury history, playoff performance, or fan engagement metrics.
""")
