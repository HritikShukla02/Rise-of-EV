"""
Complete Interrupted Time Series Analysis (ITSA) for EV Policy Impact
Author: Research Analysis Template
Date: 2025

This script performs causal impact analysis of policy interventions on EV adoption in India.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION: DEFINE KEY INTERVENTION DATES
# ============================================================================

INTERVENTIONS = {
    'FAME_II': {
        'date': '2019-04-01',
        'description': 'FAME II Scheme',
        'type': 'policy',
        'color': '#E74C3C'
    },
    'Product_Launch': {
        'date': '2020-01-20',
        'description': 'Major Product Launches',
        'type': 'product',
        'color': '#3498DB'
    },
    'COVID': {
        'date': '2020-03-25',
        'description': 'COVID-19 Lockdown',
        'type': 'shock',
        'color': '#95A5A6'
    },
    'PLI': {
        'date': '2021-09-01',
        'description': 'PLI Scheme',
        'type': 'policy',
        'color': '#2ECC71'
    }
}

# ============================================================================
# PART 1: DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_itsa_data(df, date_col='date', outcome_col='registrations'):
    """
    Prepare data for Interrupted Time Series Analysis
    
    Parameters:
    -----------
    df : DataFrame with time series data
    date_col : name of date column
    outcome_col : name of outcome variable
    
    Returns:
    --------
    DataFrame with time index added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Create time index (months since start)
    df['time'] = (df[date_col] - df[date_col].min()).dt.days / 30.44
    df['time_int'] = range(len(df))
    
    return df


def add_intervention_variables(df, intervention_date, intervention_name, date_col='date'):
    """
    Add intervention dummy and interaction terms for ITSA
    
    Model variables:
    - {name}_post: Binary indicator (0 before, 1 after)
    - {name}_time_since: Time since intervention (0 before)
    """
    df = df.copy()
    intervention_date = pd.to_datetime(intervention_date)
    
    # Binary indicator
    df[f'{intervention_name}_post'] = (df[date_col] >= intervention_date).astype(int)
    
    # Time since intervention
    df[f'{intervention_name}_time_since'] = df.apply(
        lambda row: (row[date_col] - intervention_date).days / 30.44 
        if row[f'{intervention_name}_post'] == 1 else 0,
        axis=1
    )
    
    return df


def add_all_interventions(df, date_col='date'):
    """Add all intervention variables from INTERVENTIONS dict"""
    df_result = df.copy()
    for interv_name, interv_info in INTERVENTIONS.items():
        df_result = add_intervention_variables(
            df_result, 
            interv_info['date'], 
            interv_name,
            date_col
        )
    return df_result


# ============================================================================
# PART 2: SINGLE INTERVENTION ITSA
# ============================================================================

def run_single_intervention_itsa(df, intervention_name, outcome='registrations', 
                                  log_transform=True):
    """
    Run ITSA for a single intervention
    
    Model: Y = β0 + β1*time + β2*intervention + β3*time_since + ε
    
    β0 = baseline level
    β1 = baseline trend (pre-intervention slope)
    β2 = immediate level change (step change)
    β3 = change in trend (difference in slopes)
    
    Returns:
    --------
    dict with model, data, and causal effects
    """
    df = df.copy()
    
    # Prepare outcome variable
    if log_transform and (df[outcome] > 0).all():
        df['outcome'] = np.log(df[outcome] + 1)
        outcome_label = f'log({outcome})'
        is_log = True
    else:
        df['outcome'] = df[outcome]
        outcome_label = outcome
        is_log = False
    
    # Build formula
    post_var = f'{intervention_name}_post'
    time_since_var = f'{intervention_name}_time_since'
    formula = f'outcome ~ time + {post_var} + {time_since_var}'
    
    # Fit OLS model
    model = smf.ols(formula, data=df).fit()
    
    # Predictions
    df['predicted'] = model.predict(df)
    
    # Counterfactual (set intervention to 0)
    df_counterfactual = df.copy()
    df_counterfactual[post_var] = 0
    df_counterfactual[time_since_var] = 0
    df['counterfactual'] = model.predict(df_counterfactual)
    
    # Causal effect
    df['causal_effect'] = df['predicted'] - df['counterfactual']
    
    # Calculate cumulative effect
    post_data = df[df[post_var] == 1]
    if is_log:
        # Back-transform from log space
        cumulative = (np.exp(post_data['causal_effect']) - 1).sum()
    else:
        cumulative = post_data['causal_effect'].sum()
    
    return {
        'model': model,
        'data': df,
        'cumulative_effect': cumulative,
        'intervention_name': intervention_name,
        'outcome_label': outcome_label,
        'is_log': is_log
    }


# ============================================================================
# PART 3: MULTIPLE INTERVENTION ITSA
# ============================================================================

def run_multiple_intervention_itsa(df, outcome='registrations', log_transform=True,
                                   interventions_to_test=None):
    """
    Run ITSA with multiple interventions simultaneously
    
    Model: Y = β0 + β1*time + Σ(β2i*post_i + β3i*time_since_i) + ε
    """
    df = df.copy()
    
    if interventions_to_test is None:
        interventions_to_test = list(INTERVENTIONS.keys())
    
    # Prepare outcome
    if log_transform and (df[outcome] > 0).all():
        df['outcome'] = np.log(df[outcome] + 1)
        outcome_label = f'log({outcome})'
    else:
        df['outcome'] = df[outcome]
        outcome_label = outcome
    
    # Build formula
    formula_parts = ['outcome ~ time']
    for interv in interventions_to_test:
        post_var = f'{interv}_post'
        time_since_var = f'{interv}_time_since'
        if post_var in df.columns and time_since_var in df.columns:
            formula_parts.append(f'{post_var} + {time_since_var}')
    
    formula = ' + '.join(formula_parts)
    
    # Fit model
    model = smf.ols(formula, data=df).fit()
    
    # Extract effects
    effects = {}
    for interv in interventions_to_test:
        post_var = f'{interv}_post'
        time_since_var = f'{interv}_time_since'
        
        effects[interv] = {
            'immediate_effect': model.params.get(post_var, 0),
            'immediate_pvalue': model.pvalues.get(post_var, 1),
            'immediate_se': model.bse.get(post_var, 0),
            'trend_change': model.params.get(time_since_var, 0),
            'trend_pvalue': model.pvalues.get(time_since_var, 1),
            'trend_se': model.bse.get(time_since_var, 0)
        }
    
    return {
        'model': model,
        'effects': effects,
        'data': df,
        'outcome_label': outcome_label
    }


# ============================================================================
# PART 4: VISUALIZATION - SINGLE INTERVENTION
# ============================================================================

def plot_single_itsa(result, intervention_date, intervention_desc, 
                     figsize=(14, 6), save_path=None):
    """Create publication-quality ITSA plot for single intervention"""
    
    df = result['data']
    model = result['model']
    intervention_name = result['intervention_name']
    intervention_date = pd.to_datetime(intervention_date)
    post_var = f'{intervention_name}_post'
    time_since_var = f'{intervention_name}_time_since'
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
    fig.patch.set_facecolor('white')
    
    # LEFT: Actual vs Predicted vs Counterfactual
    ax1 = axes[0]
    
    ax1.scatter(df['date'], df['outcome'], alpha=0.6, s=50, 
                color='#34495E', label='Actual', zorder=3)
    ax1.plot(df['date'], df['predicted'], linewidth=2.5, 
             color='#E74C3C', label='Fitted', zorder=2)
    
    post_data = df[df[post_var] == 1]
    ax1.plot(post_data['date'], post_data['counterfactual'], 
             linewidth=2.5, linestyle='--', color='#3498DB',
             label='Counterfactual', zorder=2)
    
    ax1.axvline(intervention_date, color='red', linestyle=':', 
                linewidth=2, alpha=0.7, label='Intervention', zorder=1)
    
    ax1.set_xlabel('Date', fontsize=11, fontweight='medium')
    ax1.set_ylabel(result['outcome_label'], fontsize=11, fontweight='medium')
    ax1.set_title(f'(a) {intervention_desc}', fontsize=12, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=9, frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # RIGHT: Causal Effect
    ax2 = axes[1]
    
    effect_data = df[df[post_var] == 1].copy()
    ax2.fill_between(effect_data['date'], 0, effect_data['causal_effect'],
                     alpha=0.3, color='#2ECC71', label='Causal effect')
    ax2.plot(effect_data['date'], effect_data['causal_effect'],
             linewidth=2.5, color='#27AE60', marker='o', markersize=4)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.axvline(intervention_date, color='red', linestyle=':', 
                linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=11, fontweight='medium')
    ax2.set_ylabel('Causal Effect', fontsize=11, fontweight='medium')
    ax2.set_title(f'(b) Estimated Impact', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Statistics box
    immediate = model.params[post_var]
    immediate_p = model.pvalues[post_var]
    trend = model.params[time_since_var]
    trend_p = model.pvalues[time_since_var]
    
    stats_text = f'Immediate: {immediate:.3f} (p={immediate_p:.4f})\n'
    stats_text += f'Trend: {trend:.3f} (p={trend_p:.4f})\n'
    stats_text += f'R² = {model.rsquared:.3f}'
    
    ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes,
             fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# PART 5: VISUALIZATION - COEFFICIENT COMPARISON
# ============================================================================

def plot_coefficient_comparison(multi_result, figsize=(12, 6), save_path=None):
    """Create coefficient plot comparing all interventions"""
    
    effects = multi_result['effects']
    model = multi_result['model']
    
    # Prepare data
    interventions = []
    immediate_effects = []
    immediate_ci_lower = []
    immediate_ci_upper = []
    immediate_colors = []
    trend_effects = []
    trend_ci_lower = []
    trend_ci_upper = []
    trend_colors = []
    
    for interv_name, effect_dict in effects.items():
        interventions.append(INTERVENTIONS[interv_name]['description'])
        
        # Immediate effects
        imm_eff = effect_dict['immediate_effect']
        imm_se = effect_dict['immediate_se']
        immediate_effects.append(imm_eff)
        immediate_ci_lower.append(imm_eff - 1.96*imm_se)
        immediate_ci_upper.append(imm_eff + 1.96*imm_se)
        immediate_colors.append('#E74C3C' if effect_dict['immediate_pvalue'] < 0.05 else '#BDC3C7')
        
        # Trend changes
        trend_eff = effect_dict['trend_change']
        trend_se = effect_dict['trend_se']
        trend_effects.append(trend_eff)
        trend_ci_lower.append(trend_eff - 1.96*trend_se)
        trend_ci_upper.append(trend_eff + 1.96*trend_se)
        trend_colors.append('#3498DB' if effect_dict['trend_pvalue'] < 0.05 else '#BDC3C7')
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(interventions))
    
    # LEFT: Immediate Effects
    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, immediate_effects, color=immediate_colors, alpha=0.7)
    ax1.errorbar(immediate_effects, y_pos,
                 xerr=[np.array(immediate_effects) - np.array(immediate_ci_lower),
                       np.array(immediate_ci_upper) - np.array(immediate_effects)],
                 fmt='none', color='black', capsize=5, linewidth=2)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(interventions, fontsize=10)
    ax1.set_xlabel('Immediate Effect Coefficient', fontsize=11, fontweight='medium')
    ax1.set_title('(a) Immediate Level Change', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add significance stars
    for i, (interv_name, effect) in enumerate(effects.items()):
        p = effect['immediate_pvalue']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        x_pos = immediate_effects[i] + 0.01 * (max(immediate_effects) - min(immediate_effects))
        ax1.text(x_pos, i, f' {stars}', va='center', fontsize=9, fontweight='bold')
    
    # RIGHT: Trend Changes
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, trend_effects, color=trend_colors, alpha=0.7)
    ax2.errorbar(trend_effects, y_pos,
                 xerr=[np.array(trend_effects) - np.array(trend_ci_lower),
                       np.array(trend_ci_upper) - np.array(trend_effects)],
                 fmt='none', color='black', capsize=5, linewidth=2)
    
    ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(interventions, fontsize=10)
    ax2.set_xlabel('Trend Change Coefficient', fontsize=11, fontweight='medium')
    ax2.set_title('(b) Change in Slope', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add significance stars
    for i, (interv_name, effect) in enumerate(effects.items()):
        p = effect['trend_pvalue']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        x_pos = trend_effects[i] + 0.01 * (max(trend_effects) - min(trend_effects))
        ax2.text(x_pos, i, f' {stars}', va='center', fontsize=9, fontweight='bold')
    
    # Legend
    legend_text = '*** p<0.001, ** p<0.01, * p<0.05, ns=not significant'
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# PART 6: DIAGNOSTIC TESTS
# ============================================================================

def run_diagnostic_tests(model, data):
    """Run diagnostic tests for model assumptions"""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TESTS")
    print("="*70)
    
    # 1. Durbin-Watson (autocorrelation)
    dw_stat = durbin_watson(model.resid)
    print(f"\n1. Durbin-Watson: {dw_stat:.3f}")
    print(f"   {'✓ No autocorrelation' if 1.5 <= dw_stat <= 2.5 else '⚠ Possible autocorrelation'}")
    
    # 2. Breusch-Pagan (heteroscedasticity)
    try:
        _, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
        print(f"\n2. Breusch-Pagan p-value: {bp_pvalue:.4f}")
        print(f"   {'✓ Homoscedastic' if bp_pvalue > 0.05 else '⚠ Heteroscedastic'}")
    except:
        print("\n2. Breusch-Pagan: Could not compute")
    
    # 3. Shapiro-Wilk (normality)
    if len(model.resid) < 5000:  # Shapiro-Wilk has sample size limit
        _, sw_pvalue = stats.shapiro(model.resid)
        print(f"\n3. Shapiro-Wilk p-value: {sw_pvalue:.4f}")
        print(f"   {'✓ Normal residuals' if sw_pvalue > 0.05 else '⚠ Non-normal residuals'}")
    
    # 4. Cook's Distance (outliers)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(data)
    outliers = np.where(cooks_d > threshold)[0]
    print(f"\n4. Influential outliers: {len(outliers)}")
    if len(outliers) > 0 and len(outliers) < 10:
        print(f"   Indices: {outliers}")
    
    print("="*70)


# ============================================================================
# PART 7: RESULTS SUMMARY
# ============================================================================

def print_results_summary(result, intervention_name, intervention_desc):
    """Print formatted results summary"""
    
    model = result['model']
    post_var = f'{intervention_name}_post'
    time_var = f'{intervention_name}_time_since'
    
    print("\n" + "="*70)
    print(f"RESULTS: {intervention_desc}")
    print("="*70)
    
    print(f"\nModel: {result['outcome_label']} ~ time + intervention + time_since")
    print(f"R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
    print(f"F-statistic = {model.fvalue:.2f}, p = {model.f_pvalue:.4e}")
    
    print("\n--- COEFFICIENTS ---")
    print(f"Baseline trend: {model.params['time']:.4f} (p={model.pvalues['time']:.4f})")
    print(f"Immediate effect: {model.params[post_var]:.4f} (p={model.pvalues[post_var]:.4f})")
    print(f"Trend change: {model.params[time_var]:.4f} (p={model.pvalues[time_var]:.4f})")
    
    # Interpretation
    print("\n--- INTERPRETATION ---")
    immediate = model.params[post_var]
    immediate_sig = "significant" if model.pvalues[post_var] < 0.05 else "not significant"
    trend = model.params[time_var]
    trend_sig = "significant" if model.pvalues[time_var] < 0.05 else "not significant"
    
    print(f"• Immediate change: {abs(immediate):.3f} ({'increase' if immediate > 0 else 'decrease'}), {immediate_sig}")
    print(f"• Trend change: {abs(trend):.4f}/month ({'acceleration' if trend > 0 else 'deceleration'}), {trend_sig}")
    print(f"• Cumulative effect: {result['cumulative_effect']:.2f}")
    
    print("="*70 + "\n")


# ============================================================================
# PART 8: MAIN ANALYSIS PIPELINE
# ============================================================================

def main_analysis_pipeline(df, outcome='EV + Hybrid', date_col='date'):
    """
    Complete ITSA analysis pipeline
    
    Parameters:
    -----------
    df : DataFrame with time series data
         Must have columns: [date_col, outcome]
    outcome : str, name of outcome variable
    date_col : str, name of date column
    
    Returns:
    --------
    dict with all results
    """
    
    print("\n" + "="*70)
    print("INTERRUPTED TIME SERIES ANALYSIS (ITSA)")
    print("EV Policy Impact Evaluation")
    print("="*70)
    
    # Validate input
    if outcome not in df.columns:
        raise ValueError(f"Column '{outcome}' not found in dataframe")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe")
    
    # Step 1: Prepare data
    print("\n[1/5] Preparing data...")
    df_prepared = prepare_itsa_data(df, date_col=date_col, outcome_col=outcome)
    df_prepared = add_all_interventions(df_prepared, date_col=date_col)
    print(f"✓ Data prepared: {len(df_prepared)} observations")
    
    # Step 2: Individual analyses
    print("\n[2/5] Running individual intervention analyses...")
    individual_results = {}
    
    for interv_name, interv_info in INTERVENTIONS.items():
        print(f"\n  → {interv_info['description']}")
        
        result = run_single_intervention_itsa(
            df_prepared, 
            interv_name, 
            outcome=outcome,
            log_transform=True
        )
        individual_results[interv_name] = result
        
        print_results_summary(result, interv_name, interv_info['description'])
        
        plot_single_itsa(
            result, 
            interv_info['date'], 
            interv_info['description'],
            save_path=f'itsa_{interv_name}.png'
        )
        
        run_diagnostic_tests(result['model'], result['data'])
    
    # Step 3: Combined model
    print("\n[3/5] Running combined model with all interventions...")
    multi_result = run_multiple_intervention_itsa(
        df_prepared,
        outcome=outcome,
        log_transform=True
    )
    
    print("\n" + "="*70)
    print("COMBINED MODEL RESULTS")
    print("="*70)
    print(f"R² = {multi_result['model'].rsquared:.4f}")
    print(f"Adj. R² = {multi_result['model'].rsquared_adj:.4f}\n")
    
    for interv_name, effects in multi_result['effects'].items():
        print(f"{INTERVENTIONS[interv_name]['description']}:")
        print(f"  Immediate: {effects['immediate_effect']:.4f} (p={effects['immediate_pvalue']:.4f})")
        print(f"  Trend: {effects['trend_change']:.4f} (p={effects['trend_pvalue']:.4f})")
    
    # Step 4: Comparison plot
    print("\n[4/5] Creating comparison plot...")
    plot_coefficient_comparison(multi_result, save_path='itsa_comparison.png')
    
    # Step 5: Model diagnostics
    print("\n[5/5] Running diagnostics on combined model...")
    run_diagnostic_tests(multi_result['model'], multi_result['data'])
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    for interv_name in INTERVENTIONS.keys():
        print(f"  • itsa_{interv_name}.png")
    print("  • itsa_comparison.png")
    print("="*70)
    
    return {
        'individual_results': individual_results,
        'combined_result': multi_result,
        'data': df_prepared
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ITSA ANALYSIS TEMPLATE - READY TO USE")
    print("="*70)
    print("\nQuick Start:")
    print("-" * 70)
    print("""
# 1. Load your data
df = pd.read_csv('your_ev_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 2. Run analysis
results = main_analysis_pipeline(df, outcome='EV + Hybrid')

# 3. Access results
fame_result = results['individual_results']['FAME_II']
combined_model = results['combined_result']['model']
print(combined_model.summary())
    """)
    print("-" * 70)
    print("\nRequired data format:")
    print("  • Column 'date': datetime format")
    print("  • Column for outcome (e.g., 'EV + Hybrid'): numeric")
    print("  • At least 20-30 time points for robust analysis")
    print("="*70)
