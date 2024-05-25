# README: Best European Football League Analysis

## Project Overview

This project aims to determine the best European football league over the past five seasons using a data-driven approach. The leagues analyzed include Bundesliga, Ligue 1, Serie A, and Premier League. The analysis involves setting up an ETL (Extract, Transform, Load) pipeline for data collection and processing, and developing an AI/ML model to analyze league statistics.

## Data Collection

Data was collected from the official websites of the respective leagues:

- **Bundesliga**: [bundesliga.com](https://www.bundesliga.com/en/bundesliga/table)
- **Ligue 1**: [ligue1.com](https://www.ligue1.com/ranking)
- **Serie A**: [legaseriea.it](https://www.legaseriea.it/en/serie-a/classifica)
- **Premier League**: [premierleague.com](https://www.premierleague.com/tables)

Each league's data for the past five seasons (2019-2020 to 2023-2024) was fetched using API calls and web scraping techniques. The collected data included team standings, goals scored, goals conceded, points, and other relevant statistics.

## Data Storage

The collected data is stored in CSV files, organized by league and season. This structured approach ensures easy access and management of data for subsequent processing.

## Data Processing

Data processing involves cleaning and transforming the raw data to make it suitable for analysis. This includes:

- Handling missing values
- Normalizing data for comparison
- Calculating additional metrics such as goal difference and average points per team

Processed data is stored separately to ensure the integrity and reproducibility of the analysis.

## AI Model

To determine the best league, several metrics for each league were calculated:

- Total goals scored
- Total goals conceded
- Average points per team
- Standard deviation of points

Normalization of these metrics allowed for their aggregation into a single score. The league with the highest aggregate score is considered the best.

## Python Script

The Python script provided (`4mac.py`) includes the complete code used for data collection, processing, and analysis. Here is an outline of the script:

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

# Function to load CSV files
def load_data(file_path):
    return pd.read_csv(file_path)

# Define the path where your datasets are stored
dataset_path = '/content/drive/MyDrive/Colab Notebooks/4MAC/datasets/'

# List of leagues and seasons
leagues = ['bundesliga', 'ligue1', 'serieA', 'premierLeague']
seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# Load the data
data_frames = {}
for league in leagues:
    data_frames[league] = {}
    for season in seasons:
        file_name = f'{league}/{season}.csv'
        data_frames[league][season] = load_data(os.path.join(dataset_path, file_name))

# Process each DataFrame
league_stats = []

for league, season_data in data_frames.items():
    total_goals_scored = 0
    total_goals_conceded = 0
    total_points = 0
    total_teams = 0
    all_points = []
    
    for season, df in season_data.items():
        total_goals_scored += df['Goals For'].sum()
        total_goals_conceded += df['Goals Against'].sum()
        total_points += df['Points'].sum()
        total_teams += len(df)
        all_points.extend(df['Points'].values)
    
    average_points_per_team = total_points / total_teams
    std_dev_points = np.std(all_points)
    
    league_stats.append({
        'league': league,
        'total_goals_scored': total_goals_scored,
        'total_goals_conceded': total_goals_conceded,
        'average_points_per_team': average_points_per_team,
        'std_dev_points': std_dev_points
    })

# Convert to DataFrame
df_league_stats = pd.DataFrame(league_stats)

# Normalization
df_league_stats['total_goals_scored_norm'] = (df_league_stats['total_goals_scored'] - df_league_stats['total_goals_scored'].min()) / (df_league_stats['total_goals_scored'].max() - df_league_stats['total_goals_scored'].min())
df_league_stats['total_goals_conceded_norm'] = (df_league_stats['total_goals_conceded'].min() - df_league_stats['total_goals_conceded']) / (df_league_stats['total_goals_conceded'].min() - df_league_stats['total_goals_conceded'].max())
df_league_stats['average_points_per_team_norm'] = (df_league_stats['average_points_per_team'] - df_league_stats['average_points_per_team'].min()) / (df_league_stats['average_points_per_team'].max() - df_league_stats['average_points_per_team'].min())
df_league_stats['std_dev_points_norm'] = (df_league_stats['std_dev_points'].min() - df_league_stats['std_dev_points']) / (df_league_stats['std_dev_points'].min() - df_league_stats['std_dev_points'].max())

# Aggregate score (assuming equal weight for simplicity)
df_league_stats['aggregate_score'] = (df_league_stats['total_goals_scored_norm'] + df_league_stats['total_goals_conceded_norm'] + df_league_stats['average_points_per_team_norm'] + df_league_stats['std_dev_points_norm']) / 4

# Determine the best league
best_league = df_league_stats.loc[df_league_stats['aggregate_score'].idxmax(), 'league']

print("The best league based on the given criteria is:", best_league)

# Cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# Create empty list to store the scores
scores = []

for train_index, test_index in kf.split(df_league_stats):
    train, test = df_league_stats.iloc[train_index], df_league_stats.iloc[test_index]
    best_league_train = train.loc[train['aggregate_score'].idxmax(), 'league']
    best_league_test = test.loc[test['aggregate_score'].idxmax(), 'league']
    score = best_league_train == best_league_test
    scores.append(score)

print("Cross-validation accuracy:", np.mean(scores))

# Visualizations
# Boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_league_stats[['total_goals_scored_norm', 'total_goals_conceded_norm', 'average_points_per_team_norm', 'std_dev_points_norm']])
plt.title('Boxplot of Normalized Metrics')
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_league_stats.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.show()
```

## Results

The Premier League was determined to be the best league based on the aggregate score. However, the cross-validation accuracy was low, indicating variability in the league rankings when different splits of data were used for training and testing. This suggests that while the Premier League performed well on average, there is significant competition among the top leagues.

## Visualization

Visualizations were created to provide insights into the data:

1. **Boxplot**: Displayed the distribution of normalized metrics for each league.
2. **Heatmap**: Showed the correlations between different metrics.

## Conclusion

The analysis provided a quantitative method to determine the best European football league based on key performance metrics over the last five seasons. While the Premier League emerged as the best league in this analysis, the results also highlighted the competitive nature of European football leagues.

For further improvement, more sophisticated models and additional data sources could be incorporated to enhance the robustness of the analysis.

This README provides a comprehensive guide to understanding the project. For any questions or further assistance, please refer to the provided documentation or contact the project maintainers.
