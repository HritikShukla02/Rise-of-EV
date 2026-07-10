"""
Complete State-Level Panel ITSA Analysis
Addresses spatial heterogeneity and aggregation bias in policy evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.diagnostic import het_breuschpagan
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# PART 1: PREPARE STATE-LEVEL PANEL DATA
# ============================================================================

def prepare_state_panel_data(df, state_col='STATE', year_col='YEAR'):
    """
    Prepare state-level panel data for ITSA
    
    Returns:
    --------
    DataFrame with log-transformed outcomes and time variables
    """
    df = df.copy()
    
    # Calculate EV variables
    df['EV + Hybrid'] = df['Electric (EV)'] + df['Hybrid EV']
    df['EV_market_share'] = (df['EV + Hybrid'] / df['TOTAL']) * 100
    
    # Create time variable
    df['time'] = df[year_col] - df[year_col].min()
    
    # Log transform (add 1 to handle zeros)
    df['log_ev'] = np.log(df['EV + Hybrid'] + 1)
    df['log_total'] = np.log(df['TOTAL'])
    
    # Sort by state and year
    df = df.sort_values([state_col, year_col]).reset_index(drop=True)
    
    print(f"✓ Prepared panel: {df[state_col].nunique()} states × {df[year_col].nunique()} years")
    
    return df


def add_state_interventions(df, state_col='STATE', year_col='YEAR'):
    """
    Add intervention variables for national and state-specific policies
    """
    df = df.copy()
    
    year_min = df[year_col].min()
    
    # National interventions (apply to ALL states)
    interventions = {
        'FAME_II': 2019,
        'Product_Launch': 2020,
        'COVID': 2020,
        'PLI': 2021
    }
    
    for name, year in interventions.items():
        df[name] = (df[year_col] >= year).astype(int)
        df[f'{name}_time'] = df.apply(
            lambda x: x['time'] - (year - year_min) if x[name] == 1 else 0,
            axis=1
        )
    
    # State-specific policies (customize based on actual policy dates)
    state_policies = {
        'DELHI': 2020,           # Delhi EV Policy
        'GUJARAT': 2021,         # Gujarat EV Policy
        'MAHARASHTRA': 2021,     # Maharashtra EV Policy
        'KARNATAKA': 2017,       # Karnataka EV Policy (early adopter)
        'TAMIL NADU': 2019,      # Tamil Nadu EV Policy
    }
    
    for state, year in state_policies.items():
        col_name = f'{state}_Policy'
        df[col_name] = ((df[state_col] == state) & (df[year_col] >= year)).astype(int)
    
    return df


# ============================================================================
# PART 2: PANEL FIXED EFFECTS ITSA
# ============================================================================

def run_panel_itsa_fixed_effects(df, outcome='log_ev', state_col='STATE'):
    """
    Run panel ITSA with state fixed effects and cluster-robust SEs
    
    Model: Y_st = α_s + β₁*time + β₂*intervention + β₃*time_since + ε_st
    """
    
    # Create formula with all interventions
    formula = (f'{outcome} ~ time + '
               f'FAME_II + FAME_II_time + '
               f'Product_Launch + Product_Launch_time + '
               f'COVID + COVID_time + '
               f'PLI + PLI_time + '
               f'C({state_col})')
    
    # Fit OLS with state dummies (fixed effects)
    model = ols(formula, data=df).fit()
    
    # Cluster-robust standard errors by state
    model_clustered = model.get_robustcov_results(cov_type='cluster', 
                                                   groups=df[state_col])
    
    return {
        'model': model_clustered,
        'model_base': model,
        'formula': formula
    }


def run_panel_itsa_random_effects(df, outcome='log_ev', state_col='STATE'):
    """
    Run panel ITSA with random effects (mixed linear model)
    
    Better when number of states is large relative to observations per state
    """
    
    formula = (f'{outcome} ~ time + '
               f'FAME_II + FAME_II_time + '
               f'Product_Launch + Product_Launch_time + '
               f'COVID + COVID_time + '
               f'PLI + PLI_time')
    
    try:
        # Fit mixed model with random intercepts by state
        model = mixedlm(formula, data=df, groups=df[state_col]).fit(method='powell')
        
        return {
            'model': model,
            'formula': formula,
            'converged': True
        }
    except Exception as e:
        print(f"  ⚠ Random effects model failed: {str(e)}")
        print(f"  → Using pooled OLS instead (no random effects)")
        
        # Fallback: pooled OLS without state effects
        model = ols(formula, data=df).fit()
        
        return {
            'model': model,
            'formula': formula,
            'converged': False
        }


# ============================================================================
# PART 3: HETEROGENEOUS TREATMENT EFFECTS
# ============================================================================

def analyze_heterogeneous_effects(df, state_col='STATE', outcome='log_ev'):
    """
    Test if intervention effects vary by state characteristics
    
    Uses median split on baseline adoption to create high/low groups
    """
    
    # Get baseline (pre-2019) average EV adoption by state
    baseline_data = df[df['YEAR'] < 2019].copy()
    state_baseline = baseline_data.groupby(state_col)['EV + Hybrid'].mean()
    
    # Split at median
    median_adoption = state_baseline.median()
    high_adoption_states = state_baseline[state_baseline > median_adoption].index
    
    df['High_Adoption_State'] = df[state_col].isin(high_adoption_states).astype(int)
    
    # Create interaction terms
    df['FAME_II_x_High'] = df['FAME_II'] * df['High_Adoption_State']
    df['Product_Launch_x_High'] = df['Product_Launch'] * df['High_Adoption_State']
    df['PLI_x_High'] = df['PLI'] * df['High_Adoption_State']
    
    # Model with interactions
    formula = (f'{outcome} ~ time + '
               f'FAME_II + FAME_II_time + FAME_II_x_High + '
               f'Product_Launch + Product_Launch_time + Product_Launch_x_High + '
               f'COVID + COVID_time + '
               f'PLI + PLI_time + PLI_x_High + '
               f'C({state_col})')
    
    model = ols(formula, data=df).fit()
    model_robust = model.get_robustcov_results(cov_type='cluster', groups=df[state_col])
    
    return {
        'model': model_robust,
        'high_adoption_states': list(high_adoption_states),
        'low_adoption_states': list(state_baseline[state_baseline <= median_adoption].index),
        'data': df
    }


# ============================================================================
# PART 4: STATE-BY-STATE ITSA
# ============================================================================

def run_state_by_state_itsa(df, state_col='STATE', outcome='log_ev', 
                             min_observations=15):
    """
    Run separate ITSA for each state to get state-specific estimates
    """
    
    states = sorted(df[state_col].unique())
    results = {}
    
    print(f"\nRunning state-by-state ITSA for {len(states)} states...")
    
    for state in states:
        state_data = df[df[state_col] == state].copy()
        
        # Skip if insufficient data
        if len(state_data) < min_observations:
            print(f"  ⚠ Skipping {state}: only {len(state_data)} observations")
            continue
        
        formula = (f'{outcome} ~ time + '
                   f'FAME_II + FAME_II_time + '
                   f'Product_Launch + Product_Launch_time + '
                   f'COVID + COVID_time + '
                   f'PLI + PLI_time')
        
        try:
            model = ols(formula, data=state_data).fit()
            
            results[state] = {
                'model': model,
                'n_obs': len(state_data),
                'n_years': state_data['YEAR'].nunique(),
                'FAME_II_effect': model.params.get('FAME_II', np.nan),
                'FAME_II_se': model.bse.get('FAME_II', np.nan),
                'FAME_II_pval': model.pvalues.get('FAME_II', np.nan),
                'FAME_II_trend': model.params.get('FAME_II_time', np.nan),
                'Product_Launch_effect': model.params.get('Product_Launch', np.nan),
                'Product_Launch_se': model.bse.get('Product_Launch', np.nan),
                'Product_Launch_pval': model.pvalues.get('Product_Launch', np.nan),
                'Product_Launch_trend': model.params.get('Product_Launch_time', np.nan),
                'PLI_effect': model.params.get('PLI', np.nan),
                'PLI_se': model.bse.get('PLI', np.nan),
                'PLI_pval': model.pvalues.get('PLI', np.nan),
                'PLI_trend': model.params.get('PLI_time', np.nan),
                'COVID_effect': model.params.get('COVID', np.nan),
                'COVID_pval': model.pvalues.get('COVID', np.nan),
                'rsquared': model.rsquared
            }
            
        except Exception as e:
            print(f"  ⚠ Failed for {state}: {str(e)}")
            continue
    
    print(f"✓ Successfully analyzed {len(results)} states")
    
    return results


# ============================================================================
# PART 5: VISUALIZATION - FOREST PLOTS
# ============================================================================

def plot_state_heterogeneity(state_results, intervention='FAME_II', 
                            save_path=None):
    """
    Create forest plot showing state-specific intervention effects with CIs
    """
    
    # Extract data
    states = []
    effects = []
    ses = []
    pvals = []
    
    effect_key = f'{intervention}_effect'
    se_key = f'{intervention}_se'
    pval_key = f'{intervention}_pval'
    
    for state, result in state_results.items():
        if not np.isnan(result.get(effect_key, np.nan)):
            states.append(state)
            effects.append(result[effect_key])
            ses.append(result.get(se_key, 0))
            pvals.append(result[pval_key])
    
    # Convert to percentages
    effects_pct = [(np.exp(e) - 1) * 100 for e in effects]
    
    # Calculate 95% CIs
    ci_lower = [(np.exp(e - 1.96*se) - 1) * 100 for e, se in zip(effects, ses)]
    ci_upper = [(np.exp(e + 1.96*se) - 1) * 100 for e, se in zip(effects, ses)]
    
    # Sort by effect size
    sorted_idx = np.argsort(effects_pct)
    states = [states[i] for i in sorted_idx]
    effects_pct = [effects_pct[i] for i in sorted_idx]
    ci_lower = [ci_lower[i] for i in sorted_idx]
    ci_upper = [ci_upper[i] for i in sorted_idx]
    pvals = [pvals[i] for i in sorted_idx]
    
    # Colors
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in pvals]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(8, len(states)*0.3)), dpi=300)
    fig.patch.set_facecolor('white')
    
    y_pos = np.arange(len(states))
    
    # Plot bars
    ax.barh(y_pos, effects_pct, color=colors, alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    
    # Add error bars (95% CI)
    for i, (eff, lower, upper) in enumerate(zip(effects_pct, ci_lower, ci_upper)):
        ax.plot([lower, upper], [i, i], color='black', linewidth=1.5, alpha=0.8)
        ax.plot([lower, lower], [i-0.2, i+0.2], color='black', linewidth=1.5)
        ax.plot([upper, upper], [i-0.2, i+0.2], color='black', linewidth=1.5)
    
    # Reference line at zero
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(states, fontsize=9)
    ax.set_xlabel('Immediate Effect (%) with 95% CI', fontsize=11, fontweight='bold')
    ax.set_title(f'State-Level Heterogeneity: {intervention.replace("_", " ")} Impact', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance stars
    for i, (eff, pval) in enumerate(zip(effects_pct, pvals)):
        if pval < 0.05:
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            x_pos = max(effects_pct) * 0.95 if eff > 0 else min(effects_pct) * 0.95
            ax.text(x_pos, i, sig, va='center', ha='right' if eff > 0 else 'left',
                   fontsize=9, fontweight='bold', color='#E74C3C')
    
    # Legend
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='#95A5A6', alpha=0.7, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"./states_results/images/{save_path}", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def create_geographic_summary_table(state_results, intervention='FAME_II'):
    """
    Create summary table of state-level effects sorted by magnitude
    """
    
    summary_data = []
    
    effect_key = f'{intervention}_effect'
    pval_key = f'{intervention}_pval'
    trend_key = f'{intervention}_trend'
    
    for state, result in state_results.items():
        effect = result.get(effect_key, np.nan)
        pval = result.get(pval_key, np.nan)
        trend = result.get(trend_key, np.nan)
        
        if not np.isnan(effect):
            effect_pct = (np.exp(effect) - 1) * 100
            trend_pct = (np.exp(trend) - 1) * 100 if not np.isnan(trend) else np.nan
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            
            summary_data.append({
                'State': state,
                'Immediate Effect (%)': round(effect_pct, 2),
                'Trend Change (%/year)': round(trend_pct, 2) if not np.isnan(trend_pct) else 'N/A',
                'p-value': round(pval, 4),
                'Sig.': sig,
                'R²': round(result.get('rsquared', np.nan), 3),
                'N': result.get('n_obs', 0)
            })
    
    df_summary = pd.DataFrame(summary_data).sort_values('Immediate Effect (%)', ascending=False)
    
    return df_summary


# ============================================================================
# PART 6: COMPARISON VISUALIZATION
# ============================================================================

def plot_panel_vs_aggregate_comparison(panel_result, state_results):
    """
    Compare panel estimates with aggregate and state-specific estimates
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    interventions = ['FAME_II', 'Product_Launch', 'PLI']
    intervention_labels = ['FAME II', 'Product Launches', 'PLI Scheme']
    
    # Get panel model params as dict
    panel_model = panel_result['model']
    if hasattr(panel_model.params, 'index'):
        panel_params = panel_model.params.to_dict()
        panel_bse = panel_model.bse.to_dict()
    else:
        param_names = list(panel_result['model_base'].params.index)
        panel_params = dict(zip(param_names, panel_model.params))
        panel_bse = dict(zip(param_names, panel_model.bse))
    
    for idx, (interv, label) in enumerate(zip(interventions, intervention_labels)):
        ax = axes[idx]
        
        # Get panel estimate
        if interv in panel_params:
            panel_eff = (np.exp(panel_params[interv]) - 1) * 100
            panel_se = panel_bse[interv]
            panel_ci_lower = (np.exp(panel_params[interv] - 1.96*panel_se) - 1) * 100
            panel_ci_upper = (np.exp(panel_params[interv] + 1.96*panel_se) - 1) * 100
        else:
            continue
        
        # Get state-specific estimates
        state_effects = []
        for state, result in state_results.items():
            eff = result.get(f'{interv}_effect', np.nan)
            if not np.isnan(eff):
                state_effects.append((np.exp(eff) - 1) * 100)
        
        if not state_effects:
            continue
        
        # Violin plot of state effects
        parts = ax.violinplot([state_effects], positions=[0], widths=0.7,
                             showmeans=True, showextrema=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('#3498DB')
            pc.set_alpha(0.6)
        
        # Overlay panel estimate with CI
        ax.errorbar([0], [panel_eff], 
                   yerr=[[panel_eff - panel_ci_lower], [panel_ci_upper - panel_eff]],
                   fmt='D', color='#E74C3C', markersize=10, 
                   capsize=10, capthick=2, linewidth=2,
                   label='Panel Estimate')
        
        # Horizontal line at zero
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel('Immediate Effect (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {label}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([0])
        ax.set_xticklabels(['State Distribution'], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
        
        # Add statistics
        mean_state = np.mean(state_effects)
        std_state = np.std(state_effects)
        text = f'Panel: {panel_eff:+.1f}%\nState mean: {mean_state:+.1f}%\nState SD: {std_state:.1f}%'
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=8, va='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Panel Fixed Effects vs. State-Specific Estimates', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./states_results/images/panel_vs_state_comparison.png', dpi=300, 
               bbox_inches='tight', facecolor='white')
    print("✓ Saved: panel_vs_state_comparison.png")
    plt.show()


# ============================================================================
# PART 7: MAIN ANALYSIS PIPELINE
# ============================================================================

def main_state_level_analysis(df):
    """
    Complete state-level panel ITSA analysis pipeline
    """
    
    print("\n" + "="*70)
    print("STATE-LEVEL PANEL ITSA ANALYSIS")
    print("Addressing Spatial Heterogeneity in Policy Evaluation")
    print("="*70)
    
    # Step 1: Prepare data
    print("\n[1/5] Preparing state-level panel data...")
    df_panel = prepare_state_panel_data(df)
    df_panel = add_state_interventions(df_panel)
    
    n_states = df_panel['STATE'].nunique()
    n_years = df_panel['YEAR'].nunique()
    print(f"✓ Panel: {n_states} states × {n_years} years = {len(df_panel)} observations")
    
    # Step 2: Panel fixed effects
    print("\n[2/5] Running panel ITSA with state fixed effects...")
    fe_result = run_panel_itsa_fixed_effects(df_panel)
    
    print("\n" + "-"*70)
    print("PANEL FIXED EFFECTS RESULTS (Cluster-Robust SEs)")
    print("-"*70)
    print(f"R² = {fe_result['model'].rsquared:.4f}")
    print(f"Adj. R² = {fe_result['model'].rsquared_adj:.4f}")
    print(f"N = {int(fe_result['model'].nobs)}")
    
    # Get parameter names and create dict
    model = fe_result['model']
    if hasattr(model.params, 'index'):
        param_names = list(model.params.index)
        params_dict = model.params.to_dict()
        pvals_dict = model.pvalues.to_dict()
        bse_dict = model.bse.to_dict()
    else:
        # Use base model for parameter names
        param_names = list(fe_result['model_base'].params.index)
        params_dict = dict(zip(param_names, model.params))
        pvals_dict = dict(zip(param_names, model.pvalues))
        bse_dict = dict(zip(param_names, model.bse))
    
    print("\nIntervention Effects:")
    for var in ['FAME_II', 'Product_Launch', 'COVID', 'PLI']:
        if var in params_dict:
            coef = params_dict[var]
            pval = pvals_dict[var]
            se = bse_dict[var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            pct = (np.exp(coef) - 1) * 100
            print(f"  {var:20s}: {coef:7.4f} ({pct:+6.1f}%) SE={se:.4f}, p={pval:.4f} {sig}")
    
    # Step 3: Random effects comparison
    print("\n[3/5] Running panel ITSA with random effects...")
    re_result = run_panel_itsa_random_effects(df_panel)
    
    if re_result['converged']:
        print("\n" + "-"*70)
        print("PANEL RANDOM EFFECTS RESULTS")
        print("-"*70)
        print(f"Log-Likelihood = {re_result['model'].llf:.2f}")
        print(f"AIC = {re_result['model'].aic:.2f}")
        print(f"BIC = {re_result['model'].bic:.2f}")
        
        # Print random effects variance
        try:
            re_var = re_result['model'].cov_re.iloc[0, 0]
            print(f"Random Effects Variance = {re_var:.4f}")
        except:
            pass
    else:
        print("\n" + "-"*70)
        print("RANDOM EFFECTS MODEL DID NOT CONVERGE")
        print("Using pooled OLS results instead")
        print("-"*70)
        print(f"R² = {re_result['model'].rsquared:.4f}")
        print("Note: These are pooled estimates without state-specific effects")
    
    # Step 4: Heterogeneous effects
    print("\n[4/5] Analyzing heterogeneous treatment effects...")
    het_result = analyze_heterogeneous_effects(df_panel)
    
    print("\n" + "-"*70)
    print("HETEROGENEOUS EFFECTS ANALYSIS")
    print("-"*70)
    print(f"High-adoption states: {', '.join(het_result['high_adoption_states'][:5])}...")
    print(f"Low-adoption states: {', '.join(het_result['low_adoption_states'][:5])}...")
    
    # Get parameter dict for heterogeneous model
    het_model = het_result['model']
    if hasattr(het_model.params, 'index'):
        het_params = het_model.params.to_dict()
        het_pvals = het_model.pvalues.to_dict()
    else:
        param_names = list(het_result['model'].model.exog_names)
        het_params = dict(zip(param_names, het_model.params))
        het_pvals = dict(zip(param_names, het_model.pvalues))
    
    for var in ['FAME_II_x_High', 'Product_Launch_x_High', 'PLI_x_High']:
        if var in het_params:
            coef = het_params[var]
            pval = het_pvals[var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            pct = (np.exp(coef) - 1) * 100
            intervention = var.replace('_x_High', '')
            print(f"  {intervention} differential effect in high-adoption states: {pct:+.1f}% ({sig})")
    
    # Step 5: State-by-state analysis
    print("\n[5/5] Running state-by-state ITSA...")
    state_results = run_state_by_state_itsa(df_panel)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # for intervention in ['FAME_II', 'Product_Launch', 'PLI']:
    #     print(f"\nCreating forest plot for {intervention}...")
    #     plot_state_heterogeneity(state_results, intervention, 
    #                             f'state_heterogeneity_{intervention}.png')
    for intervention in ['FAME_II', 'Product_Launch', 'COVID', 'PLI']:
        print(f"\nCreating forest plot for {intervention}...")
        plot_state_heterogeneity(state_results, intervention, 
                                f'state_heterogeneity_{intervention}.png')
    # Comparison plot
    print("\nCreating panel vs. state comparison...")
    plot_panel_vs_aggregate_comparison(fe_result, state_results)
    
    # Generate summary tables
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLES")
    print("="*70)
    
    for intervention in ['FAME_II', 'Product_Launch', "COVID", 'PLI']:
        summary = create_geographic_summary_table(state_results, intervention)
        
        print(f"\n--- STATE-LEVEL EFFECTS: {intervention} ---")
        print(summary.to_string(index=False))
        
        csv_name = f'./states_results/csv/state_effects_{intervention}.csv'
        summary.to_csv(csv_name, index=False)
        print(f"✓ Saved: {csv_name}")
    
    print("\n" + "="*70)
    print("STATE-LEVEL ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  • state_heterogeneity_FAME_II.png")
    print("  • state_heterogeneity_Product_Launch.png")
    print("  • state_heterogeneity_PLI.png")
    print("  • panel_vs_state_comparison.png")
    print("  • state_effects_FAME_II.csv")
    print("  • state_effects_Product_Launch.csv")
    print("  • state_effects_PLI.csv")
    
    return {
        'panel_fe': fe_result,
        'panel_re': re_result,
        'heterogeneous': het_result,
        'state_by_state': state_results,
        'data': df_panel
    }


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         STATE-LEVEL PANEL ITSA ANALYSIS - READY TO USE              ║
╚══════════════════════════════════════════════════════════════════════╝

This analysis addresses the aggregation limitation by:

1. **Panel Fixed Effects ITSA**
   → Controls for time-invariant state characteristics
   → Uses all state-year observations for efficiency
   → Cluster-robust standard errors by state

2. **Heterogeneous Treatment Effects**
   → Tests if policies worked differently across states
   → Identifies which state characteristics predict success
   → Interaction models with baseline adoption levels

3. **State-by-State Analysis**
   → Individual effect estimates for each state
   → Forest plots with 95% confidence intervals
   → Identifies geographic patterns in policy effectiveness

4. **Visualization Suite**
   → Forest plots showing state heterogeneity
   → Comparison of panel vs. state-specific estimates
   → Summary tables with significance testing

USAGE:
------
df = pd.read_csv('state_year_registrations.csv')
results = main_state_level_analysis(df)

# Access results:
panel_model = results['panel_fe']['model']
state_estimates = results['state_by_state']
heterogeneity_test = results['heterogeneous']

# Export for paper:
panel_model.summary()  # Full regression table
    """)