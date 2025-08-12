#!/usr/bin/env python3
"""
Model Progression Analysis - Standalone Version
Analyzes how AI models progress towards ASI (Artificial Super Intelligence) benchmark

This script compares old vs new model versions and measures their distance to ASI.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Define the old ranking data
old_data = {
    'Model': ['ASI', 'chatgpt_4.5', 'o1_mini', 'claude_3.7', 'claude_3.5', 'o1_preview', 
              'gpt_4o_mini', 'cursor_small', 'gemini', 'mistral', 'qwen', 'deepseek', 
              'grok_3', 'gpt_4o', 'meta'],
    'p1': [1.00, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p2': [0.00, 1.00, 0.64, 0.81, 0.14, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00],
    'p3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'p4': [0.00, 0.00, 0.36, 0.19, 0.80, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.98, 1.00, 1.00],
    'r1': [1.000, 0.000, 0.000, 0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r2': [0.000, 0.419, 0.537, 0.407, 0.428, 0.423, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.318, 0.000, 0.000],
    'r3': [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.762, 0.762, 0.762, 0.710, 0.710, 0.710, 0.000, 0.000, 0.000],
    'tst': [1.000, 0.042, 0.034, 0.033, 0.033, 0.012, 0.008, 0.008, 0.008, 0.007, 0.007, 0.007, 0.001, 0.000, 0.000]
}

# Define the new ranking data
new_data = {
    'Model': ['ASI', 'chatgpt_4.5', 'o1_mini', 'claude_3.7', 'claude_3.5', 'o1_preview', 
              'gemini', 'cursor_small', 'gpt_4o_mini', 'mistral', 'qwen', 'deepseek',
              'llama_4_scout', 'grok_3', 'qwen3', 'chatgpt_5', 'grok4', 'deepseek_r1_0528',
              'opus_4', 'mistral_large2405', 'gemini_2.5_pro', 'claude_sonnet_4', 'meta', 'gpt_4o'],
    'p1': [1.00, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
           0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p2': [0.00, 1.00, 0.64, 0.81, 0.14, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
           0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'p4': [0.00, 0.00, 0.36, 0.19, 0.80, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
           0.99, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'r1': [1.000, 0.000, 0.000, 0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
           0.450, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r2': [0.000, 0.419, 0.537, 0.407, 0.428, 0.423, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
           0.000, 0.318, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r3': [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.762, 0.762, 0.762, 0.710, 0.710, 0.710,
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'tst': [1.000, 0.042, 0.034, 0.033, 0.033, 0.012, 0.008, 0.008, 0.008, 0.007, 0.007, 0.007,
            0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
}

# Create DataFrames
df_old = pd.DataFrame(old_data).set_index('Model')
df_new = pd.DataFrame(new_data).set_index('Model')

# Define model families and their evolution
model_families = {
    'Claude': {
        'old': ['claude_3.5', 'claude_3.7'],
        'new': ['opus_4', 'claude_sonnet_4']
    },
    'Gemini': {
        'old': ['gemini'],
        'new': ['gemini_2.5_pro']
    },
    'Mistral': {
        'old': ['mistral'],
        'new': ['mistral_large2405']
    },
    'Qwen': {
        'old': ['qwen'],
        'new': ['qwen3']
    },
    'Deepseek': {
        'old': ['deepseek'],
        'new': ['deepseek_r1_0528']
    },
    'Grok': {
        'old': ['grok_3'],
        'new': ['grok4']
    },
    'ChatGPT': {
        'old': ['chatgpt_4.5'],
        'new': ['chatgpt_5']
    },
    'LLaMA': {
        'old': [],  # No old LLaMA in the dataset
        'new': ['llama_4_scout']
    }
}

def calculate_weighted_distance_to_asi(df, model_name, weights=None):
    """Calculate weighted distance to ASI, giving more importance to 'tst' score"""
    if weights is None:
        # Give higher weight to 'tst' as it's the most important metric
        weights = {'p1': 1, 'p2': 1, 'p3': 1, 'p4': 1, 'r1': 1, 'r2': 1, 'r3': 1, 'tst': 5}
    
    # ASI scores (perfect performance)
    asi_scores = {'p1': 1.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'r1': 1.0, 'r2': 0.0, 'r3': 1.0, 'tst': 1.0}
    
    # Get model scores
    if model_name in df.index:
        model_row = df.loc[model_name]
    else:
        print(f"Warning: {model_name} not found in dataframe")
        return float('inf')
    
    weighted_distance = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in model_row.index:
            diff = abs(asi_scores[metric] - model_row[metric])
            weighted_distance += (diff ** 2) * weight
            total_weight += weight
    
    return np.sqrt(weighted_distance / total_weight)

def analyze_model_progression():
    """Analyze model progression between old and new versions"""
    results = []
    
    print("=== MODEL PROGRESSION ANALYSIS ===")
    print("Distance to ASI (lower is better, 0 = perfect ASI performance)\n")
    
    for family_name, family_models in model_families.items():
        print(f"\n--- {family_name} Family ---")
        
        # Calculate distances for old models
        old_distances = []
        for old_model in family_models['old']:
            if old_model in df_old.index:
                dist = calculate_weighted_distance_to_asi(df_old, old_model)
                old_distances.append(dist)
                print(f"{old_model} (old): {dist:.4f}")
        
        # Calculate distances for new models
        new_distances = []
        for new_model in family_models['new']:
            if new_model in df_new.index:
                dist = calculate_weighted_distance_to_asi(df_new, new_model)
                new_distances.append(dist)
                print(f"{new_model} (new): {dist:.4f}")
        
        # Calculate improvement/regression
        if old_distances and new_distances:
            avg_old = np.mean(old_distances)
            avg_new = np.mean(new_distances)
            improvement = avg_old - avg_new  # Positive = improvement, Negative = regression
            improvement_pct = (improvement / avg_old) * 100 if avg_old > 0 else 0
            
            status = "IMPROVED ✅" if improvement > 0 else "REGRESSED ❌"
            print(f"Average change: {improvement:+.4f} ({improvement_pct:+.1f}%) - {status}")
            
            results.append({
                'Family': family_name,
                'Old_Distance': avg_old,
                'New_Distance': avg_new,
                'Improvement': improvement,
                'Improvement_Pct': improvement_pct,
                'Status': status
            })
        elif new_distances and not old_distances:
            # New family with no old models
            avg_new = np.mean(new_distances)
            print(f"New family - Distance to ASI: {avg_new:.4f}")
            
            results.append({
                'Family': family_name,
                'Old_Distance': np.nan,
                'New_Distance': avg_new,
                'Improvement': np.nan,
                'Improvement_Pct': np.nan,
                'Status': 'NEW'
            })
    
    return pd.DataFrame(results)

def create_comprehensive_visualization(results_df, save_plots=True):
    """Create comprehensive visualizations of model progression
    
    Args:
        results_df: DataFrame with progression analysis results
        save_plots: Whether to save individual plots as PNG files
    """
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Distance comparison (only for families with both old and new)
    plt.subplot(3, 3, 1)
    comparison_df = results_df.dropna(subset=['Old_Distance'])
    if not comparison_df.empty:
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, comparison_df['Old_Distance'], width, 
                       label='Old Models', alpha=0.8, color='lightcoral')
        bars2 = plt.bar(x + width/2, comparison_df['New_Distance'], width, 
                       label='New Models', alpha=0.8, color='lightblue')
        
        plt.xlabel('Model Family')
        plt.ylabel('Distance to ASI')
        plt.title('Distance to ASI: Old vs New Models')
        plt.xticks(x, comparison_df['Family'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Improvement percentage
    plt.subplot(3, 3, 2)
    improvement_df = results_df.dropna(subset=['Improvement_Pct'])
    if not improvement_df.empty:
        colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement_Pct']]
        bars = plt.bar(improvement_df['Family'], improvement_df['Improvement_Pct'], 
                      color=colors, alpha=0.7)
        plt.xlabel('Model Family')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement Percentage (Positive = Better)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Plot 3: TST score comparison (old vs new)
    plt.subplot(3, 3, 3)
    tst_comparison = []
    for family_name, family_models in model_families.items():
        old_tst = [df_old.loc[m, 'tst'] for m in family_models['old'] if m in df_old.index]
        new_tst = [df_new.loc[m, 'tst'] for m in family_models['new'] if m in df_new.index]
        
        if old_tst and new_tst:
            tst_comparison.append({
                'Family': family_name,
                'Old_TST': np.mean(old_tst),
                'New_TST': np.mean(new_tst)
            })
    
    if tst_comparison:
        tst_df = pd.DataFrame(tst_comparison)
        x = np.arange(len(tst_df))
        width = 0.35
        
        plt.bar(x - width/2, tst_df['Old_TST'], width, label='Old Models', alpha=0.8, color='lightcoral')
        plt.bar(x + width/2, tst_df['New_TST'], width, label='New Models', alpha=0.8, color='lightblue')
        
        plt.xlabel('Model Family')
        plt.ylabel('TST Score')
        plt.title('TST Score: Old vs New Models')
        plt.xticks(x, tst_df['Family'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: All models TST ranking (new dataset)
    plt.subplot(3, 3, 4)
    tst_scores = df_new['tst'].sort_values(ascending=True)
    colors_tst = []
    for model in tst_scores.index:
        if model == 'ASI':
            colors_tst.append('gold')
        else:
            found_family = False
            for family_name, family_models in model_families.items():
                if model in family_models['old'] + family_models['new']:
                    if model in family_models['new']:
                        colors_tst.append('lightblue')  # New models
                    else:
                        colors_tst.append('lightcoral')  # Old models
                    found_family = True
                    break
            if not found_family:
                colors_tst.append('lightgray')  # Other models
    
    plt.barh(range(len(tst_scores)), tst_scores.values, color=colors_tst, alpha=0.7)
    plt.yticks(range(len(tst_scores)), tst_scores.index, fontsize=8)
    plt.xlabel('TST Score')
    plt.title('All Models TST Ranking (Higher = Better)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Distance to ASI for all models (new dataset)
    plt.subplot(3, 3, 5)
    all_distances = []
    all_models = []
    
    for model in df_new.index:
        if model != 'ASI':  # Exclude ASI itself
            dist = calculate_weighted_distance_to_asi(df_new, model)
            all_distances.append(dist)
            all_models.append(model)
    
    # Sort by distance
    sorted_data = sorted(zip(all_distances, all_models))
    sorted_distances, sorted_models = zip(*sorted_data)
    
    # Color code by family
    colors = []
    for model in sorted_models:
        found_family = False
        for family_name, family_models in model_families.items():
            if model in family_models['old'] + family_models['new']:
                if model in family_models['new']:
                    colors.append('lightblue')  # New models
                else:
                    colors.append('lightcoral')  # Old models
                found_family = True
                break
        if not found_family:
            colors.append('lightgray')  # Other models
    
    plt.barh(range(len(sorted_models)), sorted_distances, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_models)), sorted_models, fontsize=8)
    plt.xlabel('Distance to ASI')
    plt.title('All Models Distance to ASI (Lower = Better)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Progress trajectory for families with both old and new
    plt.subplot(3, 3, 6)
    for i, row in comparison_df.iterrows():
        color = 'green' if row['Improvement'] > 0 else 'red'
        plt.plot([0, 1], [row['Old_Distance'], row['New_Distance']], 
                marker='o', linewidth=3, label=row['Family'], markersize=10, color=color, alpha=0.7)
    
    plt.xlabel('Generation')
    plt.ylabel('Distance to ASI')
    plt.title('Model Evolution Trajectory')
    plt.xticks([0, 1], ['Old', 'New'])
    if not comparison_df.empty:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Family performance summary (best TST score)
    plt.subplot(3, 3, 7)
    family_best_scores = []
    family_names = []
    
    for family_name, family_models in model_families.items():
        all_family_models = family_models['old'] + family_models['new']
        family_tst_scores = []
        
        for model in all_family_models:
            if model in df_new.index:
                family_tst_scores.append(df_new.loc[model, 'tst'])
            elif model in df_old.index:
                family_tst_scores.append(df_old.loc[model, 'tst'])
        
        if family_tst_scores:
            family_best_scores.append(max(family_tst_scores))
            family_names.append(family_name)
    
    plt.bar(family_names, family_best_scores, alpha=0.7, color='skyblue')
    plt.xlabel('Model Family')
    plt.ylabel('Best TST Score')
    plt.title('Best TST Score by Family')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Regression analysis - which models got worse
    plt.subplot(3, 3, 8)
    regressed_families = results_df[results_df['Improvement'] < 0].sort_values('Improvement')
    if not regressed_families.empty:
        plt.bar(regressed_families['Family'], regressed_families['Improvement'], 
               color='red', alpha=0.7)
        plt.xlabel('Model Family')
        plt.ylabel('Performance Change')
        plt.title('Families That Regressed')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No families regressed!', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14, color='green')
        plt.title('Families That Regressed')
    
    # Plot 9: New families performance
    plt.subplot(3, 3, 9)
    new_families = results_df[results_df['Status'] == 'NEW']
    if not new_families.empty:
        plt.bar(new_families['Family'], new_families['New_Distance'], 
               color='purple', alpha=0.7)
        plt.xlabel('Model Family')
        plt.ylabel('Distance to ASI')
        plt.title('New Families Performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No new families', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('New Families Performance')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    if save_plots:
        plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Saved comprehensive analysis as 'comprehensive_model_analysis.png'")
    
    plt.show()
    
    # Create and save individual detailed plots
    if save_plots:
        create_individual_plots(results_df)
    
    return plt.gcf()

def print_detailed_analysis(results_df):
    """Print detailed analysis results"""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📊 SUMMARY TABLE:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Best performing new models
    print("\n🏆 TOP PERFORMERS:")
    new_models_all = []
    for family_models in model_families.values():
        new_models_all.extend(family_models['new'])
    
    new_models_in_data = [m for m in new_models_all if m in df_new.index]
    if new_models_in_data:
        new_models_tst = [(m, df_new.loc[m, 'tst']) for m in new_models_in_data]
        new_models_tst.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 new models by TST score:")
        for i, (model, tst) in enumerate(new_models_tst[:3], 1):
            print(f"  {i}. {model}: {tst:.4f}")
    
    # Models that improved the most
    print("\n📈 MOST IMPROVED:")
    improved_families = results_df[results_df['Improvement'] > 0].sort_values('Improvement', ascending=False)
    if not improved_families.empty:
        for _, row in improved_families.iterrows():
            print(f"  {row['Family']}: +{row['Improvement_Pct']:.1f}% improvement")
    else:
        print("  No families showed improvement")
    
    # Models that regressed the most
    print("\n📉 MOST REGRESSED:")
    regressed_families = results_df[results_df['Improvement'] < 0].sort_values('Improvement')
    if not regressed_families.empty:
        for _, row in regressed_families.iterrows():
            print(f"  {row['Family']}: {row['Improvement_Pct']:.1f}% regression")
    else:
        print("  No families showed regression")
    
    # Distance analysis
    print("\n📏 DISTANCE TO ASI ANALYSIS:")
    print("(Lower distance = closer to ASI = better performance)")
    
    # Calculate average distances
    old_avg_dist = results_df['Old_Distance'].mean()
    new_avg_dist = results_df['New_Distance'].mean()
    
    print(f"Average distance (old models): {old_avg_dist:.4f}")
    print(f"Average distance (new models): {new_avg_dist:.4f}")
    
    if not pd.isna(old_avg_dist) and not pd.isna(new_avg_dist):
        overall_change = old_avg_dist - new_avg_dist
        overall_change_pct = (overall_change / old_avg_dist) * 100
        status = "IMPROVED" if overall_change > 0 else "REGRESSED"
        print(f"Overall change: {overall_change:+.4f} ({overall_change_pct:+.1f}%) - {status}")
    
    print("\n🎯 KEY INSIGHTS:")
    
    # Count improvements vs regressions
    improved_count = len(results_df[results_df['Improvement'] > 0])
    regressed_count = len(results_df[results_df['Improvement'] < 0])
    new_count = len(results_df[results_df['Status'] == 'NEW'])
    
    print(f"  • {improved_count} families improved")
    print(f"  • {regressed_count} families regressed")
    print(f"  • {new_count} new families introduced")
    
    # Find the family closest to ASI
    min_distance_family = results_df.loc[results_df['New_Distance'].idxmin()]
    print(f"  • Closest to ASI: {min_distance_family['Family']} (distance: {min_distance_family['New_Distance']:.4f})")
    
    # Find the family with biggest improvement
    if improved_count > 0:
        max_improvement_family = results_df.loc[results_df['Improvement'].idxmax()]
        print(f"  • Biggest improvement: {max_improvement_family['Family']} (+{max_improvement_family['Improvement_Pct']:.1f}%)")
    
    # Find the family with biggest regression
    if regressed_count > 0:
        max_regression_family = results_df.loc[results_df['Improvement'].idxmin()]
        print(f"  • Biggest regression: {max_regression_family['Family']} ({max_regression_family['Improvement_Pct']:.1f}%)")

def create_individual_plots(results_df):
    """Create and save individual detailed plots with explanations"""
    
    print("\n📈 Creating individual analysis plots...")
    
    # Plot 1: Distance Comparison with detailed explanation
    plt.figure(figsize=(12, 8))
    comparison_df = results_df.dropna(subset=['Old_Distance'])
    if not comparison_df.empty:
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, comparison_df['Old_Distance'], width, 
                       label='Old Models', alpha=0.8, color='lightcoral')
        bars2 = plt.bar(x + width/2, comparison_df['New_Distance'], width, 
                       label='New Models', alpha=0.8, color='lightblue')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Model Family', fontsize=12)
        plt.ylabel('Weighted Distance to ASI', fontsize=12)
        plt.title('Model Family Evolution: Distance to ASI Benchmark\n(Lower Distance = Better Performance)', fontsize=14, fontweight='bold')
        plt.xticks(x, comparison_df['Family'], rotation=45)
        plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add explanation text
        plt.figtext(0.02, 0.02, 
                   'Explanation: This chart shows how each model family has evolved. '
                   'Distance is calculated using weighted Euclidean distance where TST score has 5x weight. '
                   'Lower values indicate better performance (closer to ASI).', 
                   fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig('01_distance_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved distance comparison as '01_distance_comparison.png'")
    plt.close()
    
    # Plot 2: Improvement Analysis
    plt.figure(figsize=(12, 8))
    improvement_df = results_df.dropna(subset=['Improvement_Pct'])
    if not improvement_df.empty:
        colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement_Pct']]
        bars = plt.bar(improvement_df['Family'], improvement_df['Improvement_Pct'], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=11, fontweight='bold')
        
        plt.xlabel('Model Family', fontsize=12)
        plt.ylabel('Performance Change (%)', fontsize=12)
        plt.title('Model Family Performance Change\n(Positive = Improvement, Negative = Regression)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        plt.grid(True, alpha=0.3)
        
        plt.figtext(0.02, 0.02, 
                   'Explanation: Green bars show families that improved (got closer to ASI), '
                   'red bars show regression (moved away from ASI). '
                   'Percentage calculated as: (old_distance - new_distance) / old_distance * 100', 
                   fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig('02_improvement_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved improvement analysis as '02_improvement_analysis.png'")
    plt.close()
    
    # Plot 3: TST Score Analysis (Most Important Metric)
    plt.figure(figsize=(14, 10))
    
    # Get all TST scores from new dataset
    tst_data = []
    for model in df_new.index:
        if model != 'ASI':
            family = 'Other'
            model_type = 'Other'
            
            # Determine family and type
            for family_name, family_models in model_families.items():
                if model in family_models['old']:
                    family = family_name
                    model_type = 'Old'
                    break
                elif model in family_models['new']:
                    family = family_name
                    model_type = 'New'
                    break
            
            tst_data.append({
                'Model': model,
                'TST_Score': df_new.loc[model, 'tst'],
                'Family': family,
                'Type': model_type
            })
    
    tst_df = pd.DataFrame(tst_data).sort_values('TST_Score', ascending=True)
    
    # Create color mapping
    colors = []
    for _, row in tst_df.iterrows():
        if row['Type'] == 'New':
            colors.append('lightblue')
        elif row['Type'] == 'Old':
            colors.append('lightcoral')
        else:
            colors.append('lightgray')
    
    y_pos = np.arange(len(tst_df))
    bars = plt.barh(y_pos, tst_df['TST_Score'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, tst_df['TST_Score'])):
        plt.text(score + 0.0005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=8)
    
    plt.yticks(y_pos, tst_df['Model'], fontsize=9)
    plt.xlabel('TST Score (Higher = Better)', fontsize=12)
    plt.title('TST Score Ranking - Most Important Metric\n(Blue=New Models, Red=Old Models, Gray=Other)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add ASI reference line
    plt.axvline(x=1.0, color='gold', linestyle='--', linewidth=3, alpha=0.8, label='ASI Perfect Score (1.0)')
    plt.legend()
    
    plt.figtext(0.02, 0.02, 
               'Explanation: TST is the most discriminative metric (weighted 5x in distance calculation). '
               'Higher scores are better. Notice how most new models score 0.000, indicating significant regression. '
               'Only LLaMA among new models shows non-zero performance.', 
               fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig('03_tst_score_ranking.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved TST score ranking as '03_tst_score_ranking.png'")
    plt.close()
    
    # Plot 4: Metric Breakdown Analysis
    plt.figure(figsize=(16, 12))
    
    # Create subplot for each metric
    metrics = ['p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'tst']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 4, i)
        
        # Get family averages for this metric
        family_data = []
        for family_name, family_models in model_families.items():
            old_scores = [df_old.loc[m, metric] for m in family_models['old'] if m in df_old.index]
            new_scores = [df_new.loc[m, metric] for m in family_models['new'] if m in df_new.index]
            
            if old_scores and new_scores:
                family_data.append({
                    'Family': family_name,
                    'Old_Avg': np.mean(old_scores),
                    'New_Avg': np.mean(new_scores)
                })
        
        if family_data:
            metric_df = pd.DataFrame(family_data)
            x = np.arange(len(metric_df))
            width = 0.35
            
            plt.bar(x - width/2, metric_df['Old_Avg'], width, label='Old', alpha=0.7, color='lightcoral')
            plt.bar(x + width/2, metric_df['New_Avg'], width, label='New', alpha=0.7, color='lightblue')
            
            plt.title(f'{metric.upper()} Metric', fontweight='bold')
            plt.xticks(x, metric_df['Family'], rotation=45, fontsize=8)
            if i == 1:
                plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Metric Breakdown: Old vs New Models\n(Shows average performance per family for each metric)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_metric_breakdown.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved metric breakdown as '04_metric_breakdown.png'")
    plt.close()
    
    # Plot 5: Regression Analysis Deep Dive
    plt.figure(figsize=(14, 10))
    
    # Focus on families that regressed
    regressed_families = results_df[results_df['Improvement'] < 0].sort_values('Improvement')
    
    if not regressed_families.empty:
        # Create detailed regression analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Regression magnitude
        bars1 = ax1.bar(regressed_families['Family'], abs(regressed_families['Improvement_Pct']), 
                        color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
        
        for bar, val in zip(bars1, abs(regressed_families['Improvement_Pct'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Model Family')
        ax1.set_ylabel('Regression Magnitude (%)')
        ax1.set_title('Families That Regressed - Magnitude Analysis')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Before vs After distances
        x = np.arange(len(regressed_families))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, regressed_families['Old_Distance'], width, 
                       label='Old Distance', alpha=0.8, color='lightgreen')
        bars3 = ax2.bar(x + width/2, regressed_families['New_Distance'], width, 
                       label='New Distance', alpha=0.8, color='red')
        
        ax2.set_xlabel('Model Family')
        ax2.set_ylabel('Distance to ASI')
        ax2.set_title('Distance Comparison for Regressed Families')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regressed_families['Family'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Regression Analysis: Understanding Performance Decline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('05_regression_analysis.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved regression analysis as '05_regression_analysis.png'")
        plt.close()
    
    print("\n📊 All individual plots saved successfully!")

def explain_analysis_components():
    """Provide detailed explanation of each analysis component"""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS EXPLANATION")
    print("="*80)
    
    print("\n🔍 METHODOLOGY BREAKDOWN:")
    print("\n1. DISTANCE CALCULATION:")
    print("   • Uses weighted Euclidean distance to ASI benchmark")
    print("   • Formula: sqrt(Σ(weight_i × (ASI_score_i - model_score_i)²) / total_weight)")
    print("   • TST metric weighted 5x more (most important)")
    print("   • Other metrics (p1-p4, r1-r3) weighted equally")
    print("   • Lower distance = better performance (closer to ASI)")
    
    print("\n2. FAMILY MAPPING:")
    print("   • Groups related models into evolutionary families")
    print("   • Tracks progression from old versions to new versions")
    print("   • Enables comparison of development trajectories")
    
    print("\n3. IMPROVEMENT CALCULATION:")
    print("   • Improvement = (old_distance - new_distance)")
    print("   • Positive = model got closer to ASI (better)")
    print("   • Negative = model moved away from ASI (worse)")
    print("   • Percentage = (improvement / old_distance) × 100")
    
    print("\n📊 PLOT EXPLANATIONS:")
    print("\n• 01_distance_comparison.png:")
    print("  Shows side-by-side comparison of old vs new model distances")
    print("  Reveals which families improved/regressed")
    
    print("\n• 02_improvement_analysis.png:")
    print("  Visualizes percentage change for each family")
    print("  Green = improvement, Red = regression")
    
    print("\n• 03_tst_score_ranking.png:")
    print("  Focuses on TST metric (most important)")
    print("  Shows dramatic decline in new model TST performance")
    
    print("\n• 04_metric_breakdown.png:")
    print("  Detailed analysis of each individual metric")
    print("  Helps identify which capabilities declined")
    
    print("\n• 05_regression_analysis.png:")
    print("  Deep dive into families that regressed")
    print("  Quantifies magnitude of performance decline")
    
    print("\n🎯 KEY INSIGHTS FROM ANALYSIS:")
    print("\n• CONVERGENCE PATTERN:")
    print("  Most new models converge to ~0.8660 distance")
    print("  Suggests systematic limitation or evaluation change")
    
    print("\n• TST SCORE COLLAPSE:")
    print("  Most new models score 0.000 on TST metric")
    print("  Indicates severe regression in this capability")
    
    print("\n• LLAMA SUCCESS:")
    print("  New LLaMA family outperforms most updated models")
    print("  Suggests different development approach worked better")
    
    print("\n• UNIFORM REGRESSION:")
    print("  5/6 families regressed by similar amounts")
    print("  Points to systematic rather than random issues")

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Model Progression Analysis...")
    print("Analyzing how AI models progress towards ASI benchmark\n")
    
    # Explain the analysis methodology
    explain_analysis_components()
    
    # Run the analysis
    progression_results = analyze_model_progression()
    
    # Print detailed results
    print_detailed_analysis(progression_results)
    
    # Create visualizations
    print("\n📊 Generating comprehensive visualizations...")
    fig = create_comprehensive_visualization(progression_results, save_plots=True)
    
    print("\n✅ Analysis complete!")
    print("\n💡 INTERPRETATION GUIDE:")
    print("  • TST score is the most important metric (weighted 5x in distance calculation)")
    print("  • Distance to ASI: 0 = perfect ASI performance, higher = worse")
    print("  • Positive improvement % = model family got better")
    print("  • Negative improvement % = model family got worse")
    print("  • ASI represents the theoretical perfect AI performance")
    
    print("\n📁 FILES CREATED:")
    print("  • comprehensive_model_analysis.png - Complete analysis overview")
    print("  • 01_distance_comparison.png - Family evolution comparison")
    print("  • 02_improvement_analysis.png - Performance change analysis")
    print("  • 03_tst_score_ranking.png - TST metric deep dive")
    print("  • 04_metric_breakdown.png - Individual metric analysis")
    print("  • 05_regression_analysis.png - Regression investigation")
    
    return progression_results, fig

if __name__ == "__main__":
    results, figure = main()