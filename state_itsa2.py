"""
Stratified Forest Plots - Separate Analysis by State Type
Groups states by baseline adoption, size, and variance for clearer insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: STATE CATEGORIZATION FUNCTIONS
# ============================================================================

def categorize_states(df, state_results):
    """
    Categorize states based on multiple characteristics:
    1. Baseline adoption (high/low)
    2. Market size (major/minor)
    3. Effect variance (stable/volatile)
    """
    
    # Get baseline adoption (pre-2019 average)
    baseline_data = df[df['YEAR'] < 2019].copy()
    state_baseline = baseline_data.groupby('STATE')['EV + Hybrid'].mean()
    state_total_regs = baseline_data.groupby('STATE')['TOTAL'].mean()
    
    # Calculate variance in effects across interventions
    state_variance = {}
    for state, result in state_results.items():
        effects = []
        for interv in ['FAME_II', 'Product_Launch', 'PLI']:
            eff = result.get(f'{interv}_effect', np.nan)
            if not np.isnan(eff):
                effects.append((np.exp(eff) - 1) * 100)
        state_variance[state] = np.std(effects) if len(effects) > 0 else 0
    
    # Define thresholds
    baseline_median = state_baseline.median()
    size_75th = state_total_regs.quantile(0.75)
    variance_75th = np.percentile(list(state_variance.values()), 75)
    
    # Categorize each state
    state_categories = {}
    
    for state in state_results.keys():
        baseline = state_baseline.get(state, 0)
        size = state_total_regs.get(state, 0)
        variance = state_variance.get(state, 0)
        
        # Multi-dimensional categorization
        category = {
            'state': state,
            'baseline_adoption': 'High' if baseline > baseline_median else 'Low',
            'market_size': 'Major' if size > size_75th else 'Minor',
            'volatility': 'Volatile' if variance > variance_75th else 'Stable',
            'baseline_value': baseline,
            'size_value': size,
            'variance_value': variance
        }
        
        # Create composite category
        if category['baseline_adoption'] == 'High' and category['market_size'] == 'Major':
            category['group'] = 'Established Markets'
        elif category['baseline_adoption'] == 'Low' and category['market_size'] == 'Major':
            category['group'] = 'Emerging Large Markets'
        elif category['baseline_adoption'] == 'High' and category['market_size'] == 'Minor':
            category['group'] = 'Mature Small Markets'
        else:
            category['group'] = 'Nascent Markets'
        
        state_categories[state] = category
    
    return state_categories


def get_top_states_by_metric(df, metric='TOTAL', n=10, year_range=(2020, 2024)):
    """
    Get top N states by a specific metric (for separate analysis)
    """
    recent_data = df[df['YEAR'].between(year_range[0], year_range[1])].copy()
    state_avg = recent_data.groupby('STATE')[metric].mean().sort_values(ascending=False)
    return list(state_avg.head(n).index)


# ============================================================================
# PART 2: STRATIFIED FOREST PLOT FUNCTION
# ============================================================================

def plot_stratified_forest_plot(state_results, state_categories, intervention='FAME_II',
                                 group_by='group', save_path=None):
    """
    Create separate forest plots for different state groups
    
    Parameters:
    -----------
    group_by : str, one of ['group', 'baseline_adoption', 'market_size', 'volatility']
    """
    
    # Extract data
    plot_data = []
    
    effect_key = f'{intervention}_effect'
    se_key = f'{intervention}_se'
    pval_key = f'{intervention}_pval'
    
    for state, result in state_results.items():
        if state not in state_categories:
            continue
            
        if not np.isnan(result.get(effect_key, np.nan)):
            effect = result[effect_key]
            se = result.get(se_key, 0)
            pval = result[pval_key]
            
            effect_pct = (np.exp(effect) - 1) * 100
            ci_lower = (np.exp(effect - 1.96*se) - 1) * 100
            ci_upper = (np.exp(effect + 1.96*se) - 1) * 100
            
            plot_data.append({
                'state': state,
                'effect': effect_pct,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'pval': pval,
                'group': state_categories[state][group_by],
                'baseline': state_categories[state]['baseline_value'],
                'size': state_categories[state]['size_value']
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Get unique groups
    groups = df_plot['group'].unique()
    n_groups = len(groups)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 10), dpi=300)
    if n_groups == 1:
        axes = [axes]
    
    fig.patch.set_facecolor('white')
    
    for idx, group in enumerate(sorted(groups)):
        ax = axes[idx]
        
        # Filter data for this group
        group_data = df_plot[df_plot['group'] == group].sort_values('effect')
        
        if len(group_data) == 0:
            continue
        
        states = group_data['state'].tolist()
        effects = group_data['effect'].tolist()
        ci_lower = group_data['ci_lower'].tolist()
        ci_upper = group_data['ci_upper'].tolist()
        pvals = group_data['pval'].tolist()
        
        y_pos = np.arange(len(states))
        
        # Colors
        colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in pvals]
        
        # Plot bars
        ax.barh(y_pos, effects, color=colors, alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        
        # Add error bars
        for i, (eff, lower, upper) in enumerate(zip(effects, ci_lower, ci_upper)):
            # Main CI line
            ax.plot([lower, upper], [i, i], color='black', linewidth=1.5, alpha=0.8)
            # End caps
            ax.plot([lower, lower], [i-0.15, i+0.15], color='black', linewidth=1.5)
            ax.plot([upper, upper], [i-0.15, i+0.15], color='black', linewidth=1.5)
        
        # Zero line
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(states, fontsize=9)
        ax.set_xlabel('Immediate Effect (%) with 95% CI', fontsize=10, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {group}\n(N={len(states)} states)', 
                     fontsize=11, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add significance stars
        for i, (eff, pval) in enumerate(zip(effects, pvals)):
            if pval < 0.05:
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                x_pos = max(effects) * 0.9 if eff > 0 else min(effects) * 0.9
                ax.text(x_pos, i, sig, va='center', 
                       ha='right' if eff > 0 else 'left',
                       fontsize=8, fontweight='bold', color='#E74C3C')
        
        # Summary stats box
        mean_effect = np.mean(effects)
        sig_count = sum(1 for p in pvals if p < 0.05)
        stats_text = f'Mean: {mean_effect:+.1f}%\nSig: {sig_count}/{len(states)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=8, va='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.4))
    
    # Overall title
    fig.suptitle(f'{intervention.replace("_", " ")} Impact by State Type', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='#95A5A6', alpha=0.7, label='Not significant')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(f"results/images/{save_path}", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# PART 3: SPECIFIC STRATIFICATION - SIZE-BASED
# ============================================================================

def plot_by_market_size(state_results, df, intervention='FAME_II', save_path=None):
    """
    Create forest plots separating major vs minor markets
    """
    
    # Get top 10 states by total registrations (2020-2024)
    recent_data = df[df['YEAR'].between(2020, 2024)].copy()
    state_sizes = recent_data.groupby('STATE')['TOTAL'].mean().sort_values(ascending=False)
    
    major_markets = list(state_sizes.head(10).index)
    
    # Separate data
    major_data = []
    minor_data = []
    
    effect_key = f'{intervention}_effect'
    se_key = f'{intervention}_se'
    pval_key = f'{intervention}_pval'
    
    for state, result in state_results.items():
        if not np.isnan(result.get(effect_key, np.nan)):
            effect = result[effect_key]
            se = result.get(se_key, 0)
            pval = result[pval_key]
            
            effect_pct = (np.exp(effect) - 1) * 100
            ci_lower = (np.exp(effect - 1.96*se) - 1) * 100
            ci_upper = (np.exp(effect + 1.96*se) - 1) * 100
            
            data_point = {
                'state': state,
                'effect': effect_pct,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'pval': pval,
                'size': state_sizes.get(state, 0)
            }
            
            if state in major_markets:
                major_data.append(data_point)
            else:
                minor_data.append(data_point)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('white')
    
    datasets = [
        (major_data, 'Major Markets (Top 10 by Total Registrations)', axes[0]),
        (minor_data, 'Minor Markets (Remaining States)', axes[1])
    ]
    
    for idx, (data, title, ax) in enumerate(datasets):
        if not data:
            continue
        
        df_group = pd.DataFrame(data).sort_values('effect')
        
        states = df_group['state'].tolist()
        effects = df_group['effect'].tolist()
        ci_lower = df_group['ci_lower'].tolist()
        ci_upper = df_group['ci_upper'].tolist()
        pvals = df_group['pval'].tolist()
        
        y_pos = np.arange(len(states))
        colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in pvals]
        
        # Plot
        ax.barh(y_pos, effects, color=colors, alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        
        # Error bars
        for i, (eff, lower, upper) in enumerate(zip(effects, ci_lower, ci_upper)):
            ax.plot([lower, upper], [i, i], color='black', linewidth=1.5, alpha=0.8)
            ax.plot([lower, lower], [i-0.15, i+0.15], color='black', linewidth=1.5)
            ax.plot([upper, upper], [i-0.15, i+0.15], color='black', linewidth=1.5)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(states, fontsize=9)
        ax.set_xlabel('Immediate Effect (%) with 95% CI', fontsize=10, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {title}\n(N={len(states)})', 
                     fontsize=11, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Significance stars
        for i, (eff, pval) in enumerate(zip(effects, pvals)):
            if pval < 0.05:
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                x_pos = max(effects) * 0.9 if eff > 0 else min(effects) * 0.9
                ax.text(x_pos, i, sig, va='center',
                       ha='right' if eff > 0 else 'left',
                       fontsize=8, fontweight='bold', color='#E74C3C')
        
        # Stats
        mean_eff = np.mean(effects)
        sig_count = sum(1 for p in pvals if p < 0.05)
        stats_text = f'Mean: {mean_eff:+.1f}%\nSig: {sig_count}/{len(states)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.4))
    
    fig.suptitle(f'{intervention.replace("_", " ")} Impact: Major vs. Minor Markets', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='#95A5A6', alpha=0.7, label='Not significant')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(f"results/images/{save_path}", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# PART 4: BASELINE ADOPTION STRATIFICATION
# ============================================================================

def plot_by_baseline_adoption(state_results, df, intervention='FAME_II', 
                              save_path=None):
    """
    Separate high-baseline (established) vs low-baseline (nascent) markets
    """
    
    # Calculate baseline adoption (pre-2019)
    baseline_data = df[df['YEAR'] < 2019].copy()
    state_baseline = baseline_data.groupby('STATE')['EV + Hybrid'].mean()
    
    # Use median split
    median_baseline = state_baseline.median()
    high_baseline_states = state_baseline[state_baseline > median_baseline].index
    
    # Separate data
    high_baseline_data = []
    low_baseline_data = []
    
    effect_key = f'{intervention}_effect'
    se_key = f'{intervention}_se'
    pval_key = f'{intervention}_pval'
    
    for state, result in state_results.items():
        if not np.isnan(result.get(effect_key, np.nan)):
            effect = result[effect_key]
            se = result.get(se_key, 0)
            pval = result[pval_key]
            
            effect_pct = (np.exp(effect) - 1) * 100
            
            # Cap extreme values for visualization
            if abs(effect_pct) > 5000:
                effect_pct = 5000 if effect_pct > 0 else -5000
            
            ci_lower = (np.exp(effect - 1.96*se) - 1) * 100
            ci_upper = (np.exp(effect + 1.96*se) - 1) * 100
            
            # Cap CIs too
            ci_lower = max(ci_lower, -5000)
            ci_upper = min(ci_upper, 5000)
            
            data_point = {
                'state': state,
                'effect': effect_pct,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'pval': pval,
                'baseline': state_baseline.get(state, 0)
            }
            
            if state in high_baseline_states:
                high_baseline_data.append(data_point)
            else:
                low_baseline_data.append(data_point)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('white')
    
    datasets = [
        (high_baseline_data, 'Established Markets\n(High Pre-2019 Adoption)', axes[0]),
        (low_baseline_data, 'Nascent Markets\n(Low Pre-2019 Adoption)', axes[1])
    ]
    
    for idx, (data, title, ax) in enumerate(datasets):
        if not data:
            continue
        
        df_group = pd.DataFrame(data).sort_values('effect')
        
        states = df_group['state'].tolist()
        effects = df_group['effect'].tolist()
        ci_lower = df_group['ci_lower'].tolist()
        ci_upper = df_group['ci_upper'].tolist()
        pvals = df_group['pval'].tolist()
        
        y_pos = np.arange(len(states))
        colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in pvals]
        
        # Plot
        ax.barh(y_pos, effects, color=colors, alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        
        # Error bars
        for i, (eff, lower, upper) in enumerate(zip(effects, ci_lower, ci_upper)):
            ax.plot([lower, upper], [i, i], color='black', linewidth=1.5, alpha=0.8)
            ax.plot([lower, lower], [i-0.15, i+0.15], color='black', linewidth=1.5)
            ax.plot([upper, upper], [i-0.15, i+0.15], color='black', linewidth=1.5)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(states, fontsize=9)
        ax.set_xlabel('Immediate Effect (%) with 95% CI', fontsize=10, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {title}\n(N={len(states)})', 
                     fontsize=11, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Significance stars
        for i, (eff, pval) in enumerate(zip(effects, pvals)):
            if pval < 0.05:
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                x_pos = max(effects) * 0.9 if eff > 0 else min(effects) * 0.9
                ax.text(x_pos, i, sig, va='center',
                       ha='right' if eff > 0 else 'left',
                       fontsize=8, fontweight='bold', color='#E74C3C')
        
        # Stats
        mean_eff = np.mean(effects)
        median_eff = np.median(effects)
        sig_count = sum(1 for p in pvals if p < 0.05)
        stats_text = f'Mean: {mean_eff:+.1f}%\nMedian: {median_eff:+.1f}%\nSig: {sig_count}/{len(states)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.4))
    
    fig.suptitle(f'{intervention.replace("_", " ")} Impact: Established vs. Nascent Markets', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='#95A5A6', alpha=0.7, label='Not significant')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(f"results/images/{save_path}", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# PART 5: MAIN EXECUTION FUNCTION
# ============================================================================

def create_all_stratified_plots(state_results, df):
    """
    Create all stratified forest plots
    """
    
    print("\n" + "="*70)
    print("CREATING STRATIFIED FOREST PLOTS")
    print("="*70)
    
    # Categorize states
    print("\n[1/3] Categorizing states...")
    state_categories = categorize_states(df, state_results)
    
    # Print categorization summary
    groups = {}
    for state, cat in state_categories.items():
        group = cat['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(state)
    
    print("\nState Categorization:")
    for group, states in groups.items():
        print(f"  {group}: {len(states)} states")
        print(f"    {', '.join(states[:5])}{'...' if len(states) > 5 else ''}")
    
    # Generate plots
    interventions = ['FAME_II', 'Product_Launch', 'PLI']
    
    for intervention in interventions:
        print(f"\n[2/3] Creating stratified plots for {intervention}...")
        
        # By composite group
        plot_stratified_forest_plot(
            state_results, state_categories, intervention,
            group_by='group',
            save_path=f'./states_itsa/stratified_{intervention}_by_group.png'
        )
        
        # By market size
        print(f"  → By market size...")
        plot_by_market_size(
            state_results, df, intervention,
            save_path=f'./states_itsa/stratified_{intervention}_by_size.png'
        )
        
        # By baseline adoption
        print(f"  → By baseline adoption...")
        plot_by_baseline_adoption(
            state_results, df, intervention,
            save_path=f'./states_itsa/stratified_{intervention}_by_baseline.png'
        )
    
    print("\n[3/3] All stratified plots created!")
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("="*70)
    for intervention in interventions:
        print(f"\n{intervention}:")
        print(f"  • stratified_{intervention}_by_group.png")
        print(f"  • stratified_{intervention}_by_size.png")
        print(f"  • stratified_{intervention}_by_baseline.png")
    print("="*70)
    
    return state_categories


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           STRATIFIED FOREST PLOTS - READY TO USE                     ║
╚══════════════════════════════════════════════════════════════════════╝

This creates separate forest plots for different state types:

1. BY COMPOSITE GROUP (4 categories):
   • Established Markets (high adoption + major)
   • Emerging Large Markets (low adoption + major)
   • Mature Small Markets (high adoption + minor)
   • Nascent Markets (low adoption + minor)

2. BY MARKET SIZE (2 categories):
   • Major Markets (top 10 by total registrations)
   • Minor Markets (remaining states)

3. BY BASELINE ADOPTION (2 categories):
   • Established Markets (high pre-2019 adoption)
   • Nascent Markets (low pre-2019 adoption)

USAGE:
------
# After running main_state_level_analysis()
state_categories = create_all_stratified_plots(
    results['state_by_state'],
    results['data']
)

# Or create individual plots:
plot_by_market_size(state_results, df, 'FAME_II', 'fame_by_size.png')
plot_by_baseline_adoption(state_results, df, 'PLI', 'pli_by_baseline.png')
    """)