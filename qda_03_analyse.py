import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from os import path

def analyze_political_data(df):
    """
    Analyze political data containing issue and party information.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'issue' and 'party' columns
    
    Returns:
    dict: Dictionary containing various analysis results
    """
    results = {}
    
    # Valid categories
    VALID_ISSUES = ['YC', 'COL', 'H', 'EI', 'AL', 'Unknown']
    
    # 1. Basic counts and percentages
    results['issue_counts'] = df['issue'].value_counts()
    results['issue_percentages'] = df['issue'].value_counts(normalize=True) * 100
    
    results['party_counts'] = df['party'].value_counts()
    results['party_percentages'] = df['party'].value_counts(normalize=True) * 100
    
    # 2. Cross-tabulation (excluding Unknown)
    crosstab = pd.crosstab(
        df[df['party'].isin(['ALP', 'LNP'])]['party'],
        df[df['party'].isin(['ALP', 'LNP'])]['issue']
    )
    results['crosstab'] = crosstab
    
    # 3. Chi-square test for independence
    chi2, p_value, dof, expected = chi2_contingency(crosstab)
    results['chi_square_test'] = {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof
    }
    
    # 4. Normalized proportions for comparison
    normalized_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    results['normalized_crosstab'] = normalized_crosstab
    
    # 5. Calculate party-issue association strength
    # Using percentage difference from overall average for each issue
    total_proportions = df['issue'].value_counts(normalize=True)
    party_proportions = df.groupby('party')['issue'].value_counts(normalize=True).unstack()
    
    # Calculate deviation from overall proportions
    deviations = {}
    for party in ['ALP', 'LNP']:
        party_data = party_proportions.loc[party]
        deviations[party] = {
            issue: party_data[issue] - total_proportions[issue]
            for issue in VALID_ISSUES if issue != 'Unknown' and issue in party_data
        }
    results['party_issue_deviations'] = deviations
    
    return results

def print_analysis_results(results):
    """
    Print formatted analysis results.
    
    Parameters:
    results (dict): Dictionary containing analysis results
    """
    print("=== Issue Distribution ===")
    print("\nCounts:")
    print(results['issue_counts'])
    print("\nPercentages:")
    print(results['issue_percentages'].round(2))
    
    print("\n=== Party Distribution ===")
    print("\nCounts:")
    print(results['party_counts'])
    print("\nPercentages:")
    print(results['party_percentages'].round(2))
    
    print("\n=== Cross-tabulation (ALP vs LNP) ===")
    print(results['crosstab'])
    
    print("\n=== Chi-square Test Results ===")
    print(f"Chi-square statistic: {results['chi_square_test']['chi2_statistic']:.2f}")
    print(f"p-value: {results['chi_square_test']['p_value']:.4f}")
    
    print("\n=== Normalized Issue Distribution by Party (%) ===")
    print(results['normalized_crosstab'].round(2))
    
    print("\n=== Party-Issue Association Strength ===")
    print("(Percentage points deviation from overall average)")
    for party, deviations in results['party_issue_deviations'].items():
        print(f"\n{party}:")
        for issue, deviation in deviations.items():
            print(f"{issue}: {deviation*100:.2f}%")

def create_visualizations(df, results):
    """
    Create comprehensive visualizations for political data analysis using seaborn.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'issue' and 'party' columns
    results (dict): Results dictionary from analyze_political_data function
    """
    # Define issue code to full name mapping
    ISSUE_NAMES = {
        'YC': 'Youth Crime',
        'COL': 'Cost of Living',
        'H': 'Health',
        'EI': 'Energy and Infrastructure',
        'AL': 'Abortion Laws',
        'Unknown': 'Unknown'
    }
    # Set the style for all plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Issue Distribution by Party (Stacked)
    plt.subplot(2, 2, 1)
    crosstab_pct = pd.crosstab(df['party'], df['issue'], normalize='index') * 100
    # Rename columns with full names
    crosstab_pct.columns = [ISSUE_NAMES.get(col, col) for col in crosstab_pct.columns]
    sns.heatmap(crosstab_pct, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Issue Distribution by Party (%)')
    plt.xticks(rotation=45, ha='right')
    
    # 2. Party-Issue Association (Deviation from Average)
    plt.subplot(2, 2, 2)
    deviation_df = pd.DataFrame(results['party_issue_deviations']).T * 100
    # Rename both index (issues) and columns (parties)
    deviation_df.index = [ISSUE_NAMES.get(idx, idx) for idx in deviation_df.index] # type: ignore
    deviation_df.columns = [ISSUE_NAMES.get(col, col) for col in deviation_df.columns]
    sns.heatmap(deviation_df, annot=True, fmt='.1f', cmap='RdBu', center=0)
    plt.title('Party-Issue Association\n(Percentage points deviation from average)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 3. Raw Counts Distribution
    plt.subplot(2, 2, 3)
    issue_counts = pd.DataFrame({
        'Issue': [ISSUE_NAMES.get(idx, idx) for idx in results['issue_counts'].index],
        'Count': results['issue_counts'].values
    })
    sns.barplot(data=issue_counts, x='Issue', y='Count')
    plt.title('Raw Issue Counts')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Party Distribution per Issue
    plt.subplot(2, 2, 4)
    normalized_crosstab = results['normalized_crosstab'].copy()
    # Rename columns with full names
    normalized_crosstab.columns = [ISSUE_NAMES.get(col, col) for col in normalized_crosstab.columns]
    sns.heatmap(normalized_crosstab, annot=True, fmt='.1f', cmap='viridis')
    plt.title('Issue Distribution Within Parties (%)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analytics for the Election Experiment.")
    parser.add_argument("version", type=int, help="The version number for the experiment.")
    args = parser.parse_args()
    file_path = f"experiments/{args.version}/phase1/phase1_assembled.csv"
    if not path.exists(file_path):
        print(f"Error: The required file {file_path} does not exist.")
        exit(1)
    df = pd.read_csv(file_path)
    # Run analysis
    results = analyze_political_data(df)
    print_analysis_results(results)
    # Create visualizations
    create_visualizations(df, results)
    plt.savefig('analysis_results.png')
    print('\nAnalysis results saved as analysis_results.png')
    print('Analysis complete.')

