"""
Exploratory Data Analysis
=========================
Generates visualisations for the final report:
1. Class distribution (overall and by dataset)
2. Feature correlation heatmap
3. Feature distributions by class
4. Monthly fire frequency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_full_eda(df: pd.DataFrame, output_dir: str = 'figures'):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    class_counts = df['fire_occurred'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['No Fire', 'Fire'], class_counts.values, color=colors)
    axes[0].set_title('Overall Class Distribution', fontsize=13)
    axes[0].set_ylabel('Count')
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    source_fire = df.groupby(['source_dataset', 'fire_occurred']).size().unstack(fill_value=0)
    source_fire.plot(kind='bar', ax=axes[1], color=colors)
    axes[1].set_title('Class Distribution by Dataset', fontsize=13)
    axes[1].set_ylabel('Count')
    axes[1].legend(['No Fire', 'Fire'])
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    feature_cols = ['temperature', 'relative_humidity', 'wind_speed',
                    'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'fire_occurred']
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax, square=True,
                linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions by class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    features = ['temperature', 'relative_humidity', 'wind_speed',
                'rain', 'FFMC', 'DMC', 'DC', 'ISI']
    
    for ax, feat in zip(axes.flat, features):
        for label, color, name in [(0, '#2ecc71', 'No Fire'), (1, '#e74c3c', 'Fire')]:
            subset = df[df['fire_occurred'] == label][feat].dropna()
            ax.hist(subset, bins=25, alpha=0.5, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=8)
    
    plt.suptitle('Feature Distributions: Fire vs No Fire', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Monthly fire rate
    df_temp = df.copy()
    df_temp['approx_month'] = (
        np.round(np.arctan2(df_temp['month_sin'], df_temp['month_cos'])
                 * 12 / (2 * np.pi)).astype(int) % 12 + 1
    )
    
    monthly = df_temp.groupby('approx_month')['fire_occurred'].agg(['sum', 'count'])
    monthly['rate'] = monthly['sum'] / monthly['count'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bars = ax.bar(range(1, 13), monthly.reindex(range(1, 13))['rate'].fillna(0),
                  color='#e74c3c', alpha=0.7, edgecolor='#c0392b')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('Fire Occurrence Rate (%)')
    ax.set_title('Fire Occurrence Rate by Month', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/monthly_fire_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"EDA: 4 visualisations saved to {output_dir}/")


if __name__ == '__main__':
    from data_pipeline import FireDataPipeline
    
    pipeline = FireDataPipeline('data/forestfires.csv',
                                 'data/Algerian_forest_fires_dataset.csv',
                                'data/icnf_portugal_extended.csv')
    combined = pipeline.build_unified_dataset()
    run_full_eda(combined)
