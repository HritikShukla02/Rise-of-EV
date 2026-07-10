"""
Enhanced Interrupted Time Series Analysis (ITSA) for EV Policy Impact
Addresses autocorrelation, heteroscedasticity, and provides robust inference

Key improvements:
- Newey-West HAC standard errors
- ARIMAX models for autocorrelation
- Weighted Least Squares for heteroscedasticity
- Enhanced diagnostics with ACF/PACF plots
- Lagged dependent variables option
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
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
# PART 1: DATA PREPARATION (unchanged)
# ============================================================================

def prepare_itsa_data(df, date_col='date', outcome_col='registrations'):
    """Prepare data for ITSA"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    df['time'] = (df[date_col] - df[date_col].min()).dt.days / 30.44
    df['time_int'] = range(len(df))
    
    return df


def add_intervention_variables(df, intervention_date, intervention_name, date_col='date'):
    """Add intervention dummy and interaction terms"""
    df = df.copy()
    intervention_date = pd.to_datetime(intervention_date)
    
    df[f'{intervention_name}_post'] = (df[date_col] >= intervention_date).astype(int)
    
    df[f'{intervention_name}_time_since'] = df.apply(
        lambda row: (row[date_col] - intervention_date).days / 30.44 
        if row[f'{intervention_name}_post'] == 1 else 0,
        axis=1
    )
    
    return df


def add_all_interventions(df, date_col='date'):
    """Add all intervention variables"""
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
# PART 2: ROBUST ESTIMATION METHODS
# ============================================================================

def run_ols_with_newey_west(df, formula, maxlags=None):
    """
    Run OLS with Newey-West HAC standard errors
    
    This corrects for both autocorrelation AND heteroscedasticity
    without changing coefficient estimates
    """
    model = smf.ols(formula, data=df).fit()
    
    # Determine optimal lag length if not specified
    if maxlags is None:
        n = len(df)
        maxlags = int(np.floor(4 * (n/100)**(2/9)))  # Newey-West recommendation
    
    # Get Newey-West corrected results
    model_nw = model.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
    
    return {
        'model': model,
        'model_robust': model_nw,
        'maxlags': maxlags
    }


def run_wls_model(df, formula, outcome='outcome'):
    """
    Run Weighted Least Squares to handle heteroscedasticity
    
    Uses rolling variance to weight observations
    """
    # First, fit OLS to get residuals
    model_ols = smf.ols(formula, data=df).fit()
    df['residuals'] = model_ols.resid
    
    # Calculate rolling variance (weights)
    window = min(12, len(df) // 3)  # Use 12-month or 1/3 of data
    df['rolling_var'] = df['residuals'].rolling(window=window, center=True).var()
    df['rolling_var'] = df['rolling_var'].fillna(df['rolling_var'].median())
    
    # Weights are inverse of variance
    df['weights'] = 1 / np.sqrt(df['rolling_var'])
    
    # Fit WLS
    model_wls = smf.wls(formula, data=df, weights=df['weights']).fit()
    
    return {
        'model': model_wls,
        'weights': df['weights'].values
    }


def run_arimax_model(df, formula, outcome='outcome', order=(1, 0, 0)):
    """
    Run ARIMAX model (ARIMA with exogenous variables)
    
    This explicitly models autocorrelation structure
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Parse formula to get exogenous variables
    formula_parts = formula.split('~')[1].strip().split('+')
    exog_vars = [v.strip() for v in formula_parts]
    
    # Prepare exogenous matrix
    exog = df[exog_vars].values
    endog = df[outcome].values
    
    # Fit ARIMAX
    model = SARIMAX(endog, exog=exog, order=order, 
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    return {
        'model': model_fit,
        'order': order,
        'exog_vars': exog_vars
    }


def find_best_arimax_order(df, formula, outcome='outcome', max_p=3, max_q=2):
    """
    Find optimal ARIMAX order using AIC/BIC
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Parse formula
    formula_parts = formula.split('~')[1].strip().split('+')
    exog_vars = [v.strip() for v in formula_parts]
    exog = df[exog_vars].values
    endog = df[outcome].values
    
    results = []
    
    print("\n" + "="*70)
    print("SEARCHING FOR OPTIMAL ARIMAX ORDER")
    print("="*70)
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue  # Skip (0,0,0) - that's just regression
            
            try:
                model = SARIMAX(endog, exog=exog, order=(p, 0, q),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
                
                aic = model_fit.aic
                bic = model_fit.bic
                
                results.append({
                    'order': (p, 0, q),
                    'AIC': aic,
                    'BIC': bic,
                    'model': model_fit
                })
                
                print(f"ARIMA{(p,0,q)}: AIC={aic:.2f}, BIC={bic:.2f}")
                
            except:
                print(f"ARIMA{(p,0,q)}: Failed to converge")
                continue
    
    if not results:
        print("⚠ No models converged, using AR(1)")
        return (1, 0, 0)
    
    # Find best by AIC
    best_result = min(results, key=lambda x: x['AIC'])
    print(f"\n✓ Best model by AIC: ARIMA{best_result['order']}")
    print(f"  AIC: {best_result['AIC']:.2f}, BIC: {best_result['BIC']:.2f}")
    print("="*70)
    
    return best_result['order']


def run_model_with_ar_terms(df, formula, outcome='outcome', ar_lags=1):
    """
    Add lagged dependent variable to control for autocorrelation
    
    Model: Y_t = β₀ + β₁*Y_{t-1} + interventions + ε_t
    """
    df = df.copy()
    
    # Create lagged outcome
    for lag in range(1, ar_lags + 1):
        df[f'{outcome}_lag{lag}'] = df[outcome].shift(lag)
    
    # Drop rows with NaN from lagging
    df = df.dropna()
    
    # Add lagged terms to formula
    lag_terms = ' + '.join([f'{outcome}_lag{i}' for i in range(1, ar_lags + 1)])
    formula_with_lags = formula + ' + ' + lag_terms
    
    # Fit model
    model = smf.ols(formula_with_lags, data=df).fit()
    
    return {
        'model': model,
        'data': df,
        'ar_lags': ar_lags
    }


# ============================================================================
# PART 3: ENHANCED SINGLE INTERVENTION ANALYSIS
# ============================================================================

def run_robust_single_intervention_itsa(df, intervention_name, outcome='registrations',
                                         log_transform=True, method='newey-west',
                                         arimax_order='auto'):
    """
    Run ITSA with robust methods
    
    Parameters:
    -----------
    method : str, one of ['ols', 'newey-west', 'wls', 'arimax', 'ar-terms']
    arimax_order : tuple or 'auto', ARIMAX order (p,d,q). If 'auto', finds best order.
    """
    df = df.copy()
    
    # Prepare outcome
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
    
    # Choose estimation method
    if method == 'newey-west':
        result_dict = run_ols_with_newey_west(df, formula)
        model = result_dict['model_robust']
        model_base = result_dict['model']
        method_used = f"OLS with Newey-West SE (maxlags={result_dict['maxlags']})"
        
    elif method == 'wls':
        result_dict = run_wls_model(df, formula)
        model = result_dict['model']
        model_base = model
        method_used = "Weighted Least Squares"
        
    elif method == 'arimax':
        # Find best order if 'auto'
        if arimax_order == 'auto':
            best_order = find_best_arimax_order(df, formula, outcome='outcome', 
                                                max_p=3, max_q=2)
        else:
            best_order = arimax_order
        
        result_dict = run_arimax_model(df, formula, outcome='outcome', order=best_order)
        model = result_dict['model']
        model_base = None
        method_used = f"ARIMAX{best_order}"
        
    elif method == 'ar-terms':
        result_dict = run_model_with_ar_terms(df, formula, outcome='outcome', ar_lags=1)
        model = result_dict['model']
        model_base = model
        df = result_dict['data']  # Update df with lagged data
        method_used = "OLS with AR(1) term"
        
    else:  # standard OLS
        model = smf.ols(formula, data=df).fit()
        model_base = model
        method_used = "Standard OLS"
    
    # Predictions
    if method != 'arimax':
        df['predicted'] = model.predict(df)
        
        # Counterfactual
        df_counterfactual = df.copy()
        df_counterfactual[post_var] = 0
        df_counterfactual[time_since_var] = 0
        df['counterfactual'] = model.predict(df_counterfactual)
        
        df['causal_effect'] = df['predicted'] - df['counterfactual']
    else:
        # For ARIMAX, create counterfactual predictions
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        formula_parts = formula.split('~')[1].strip().split('+')
        exog_vars = [v.strip() for v in formula_parts]
        
        # Fitted values
        df['predicted'] = model.fittedvalues
        
        # Counterfactual: set intervention variables to 0
        exog_counterfactual = df[exog_vars].copy()
        exog_counterfactual[post_var] = 0
        exog_counterfactual[time_since_var] = 0
        
        # Get counterfactual predictions
        df['counterfactual'] = model.predict(exog=exog_counterfactual.values)
        df['causal_effect'] = df['predicted'] - df['counterfactual']
    
    # Cumulative effect
    post_data = df[df[post_var] == 1]
    if not post_data.empty and 'causal_effect' in df.columns:
        if is_log:
            cumulative = (np.exp(post_data['causal_effect'].dropna()) - 1).sum()
        else:
            cumulative = post_data['causal_effect'].dropna().sum()
    else:
        cumulative = np.nan
    
    return {
        'model': model,
        'model_base': model_base,
        'data': df,
        'cumulative_effect': cumulative,
        'intervention_name': intervention_name,
        'outcome_label': outcome_label,
        'is_log': is_log,
        'method': method_used
    }


# ============================================================================
# PART 5A: ARIMAX COEFFICIENT EXTRACTION HELPER
# ============================================================================

def print_arimax_coefficients(arimax_result, intervention_name):
    """Extract ARIMAX coefficients correctly - handles numpy arrays"""
    model = arimax_result['model']
    
    print("\n" + "="*70)
    print(f"ARIMAX COEFFICIENTS: {intervention_name}")
    print("="*70)
    
    params = model.params
    pvalues = model.pvalues
    bse = model.bse
    
    # Get parameter names from the model
    if hasattr(params, 'index'):
        # It's a pandas Series with named indices
        param_names = list(params.index)
        params_dict = params.to_dict()
        pvalues_dict = pvalues.to_dict()
        bse_dict = bse.to_dict()
    else:
        # It's a numpy array - need to get names from model
        try:
            param_names = list(model.param_names)
            params_dict = dict(zip(param_names, params))
            pvalues_dict = dict(zip(param_names, pvalues))
            bse_dict = dict(zip(param_names, bse))
        except:
            # Fallback: use generic indices
            param_names = [f"param_{i}" for i in range(len(params))]
            params_dict = dict(zip(param_names, params))
            pvalues_dict = dict(zip(param_names, pvalues))
            bse_dict = dict(zip(param_names, bse))
    
    print("\nAll Parameters:")
    print("-"*70)
    for name in param_names:
        coef = params_dict[name]
        se = bse_dict[name]
        pval = pvalues_dict[name]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"{name:20s}: {coef:8.4f} (SE: {se:.4f}, p={pval:.4f}) {sig}")
    
    # Extract exogenous parameters (x1, x2, x3)
    print("\n" + "="*70)
    print("INTERVENTION EFFECTS (Exogenous Variables)")
    print("="*70)
    
    var_mapping = {
        'x1': 'time (baseline trend)',
        'x2': f'{intervention_name}_post (immediate effect)',
        'x3': f'{intervention_name}_time_since (trend change)'
    }
    
    exog_found = False
    for param_name in ['x1', 'x2', 'x3']:
        if param_name in params_dict:
            exog_found = True
            coef = params_dict[param_name]
            se = bse_dict[param_name]
            pval = pvalues_dict[param_name]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            
            var_name = var_mapping.get(param_name, param_name)
            print(f"{var_name:45s}: {coef:8.4f} (SE: {se:.4f}, p={pval:.4f}) {sig}")
    
    if not exog_found:
        print("⚠ Warning: Could not find exogenous parameters (x1, x2, x3)")
        print("Available parameters:", param_names)
    
    # Interpretation
    if 'x2' in params_dict and 'x3' in params_dict:
        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("-"*70)
        
        imm_coef = params_dict['x2']
        imm_pval = pvalues_dict['x2']
        imm_sig = '***' if imm_pval < 0.001 else '**' if imm_pval < 0.01 else '*' if imm_pval < 0.05 else 'ns'
        
        trend_coef = params_dict['x3']
        trend_pval = pvalues_dict['x3']
        trend_sig = '***' if trend_pval < 0.001 else '**' if trend_pval < 0.01 else '*' if trend_pval < 0.05 else 'ns'
        
        # Calculate percentage effect (since using log transform)
        pct_change = (np.exp(imm_coef) - 1) * 100
        
        direction = "increase" if imm_coef > 0 else "decrease"
        print(f"• Immediate Effect: {imm_coef:.4f} log units = {pct_change:+.1f}% {direction} ({imm_sig})")
        
        direction = "acceleration" if trend_coef > 0 else "deceleration"
        print(f"• Trend Change: {trend_coef:.4f} log units/month ({direction}, {trend_sig})")
        
        if imm_pval > 0.05 and trend_pval < 0.05:
            print("\n→ Effect is primarily through sustained trend change, not immediate jump")
        elif imm_pval < 0.05 and trend_pval > 0.05:
            print("\n→ Effect is one-time level shift without sustained trend change")
        elif imm_pval < 0.05 and trend_pval < 0.05:
            print("\n→ Both immediate and sustained effects are significant")
        else:
            print("\n→ No significant effects detected")
    
    print("="*70)

# ============================================================================
# PART 5B: ENHANCED VISUALIZATION - SINGLE INTERVENTION
# ============================================================================

def plot_residual_diagnostics(model, data, title="", save_path=None):
    """
    Comprehensive residual diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('white')
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    # 1. Residuals over time
    ax1 = axes[0, 0]
    ax1.scatter(range(len(residuals)), residuals, alpha=0.6, s=30)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.plot(pd.Series(residuals).rolling(window=6).mean(), color='blue', linewidth=2, label='6-period MA')
    ax1.set_xlabel('Time Index', fontweight='medium')
    ax1.set_ylabel('Residuals', fontweight='medium')
    ax1.set_title('(a) Residuals Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. ACF plot
    ax2 = axes[0, 1]
    try:
        plot_acf(residuals, lags=min(20, len(residuals)//3), ax=ax2, alpha=0.05)
        ax2.set_title('(b) Autocorrelation Function', fontweight='bold')
    except:
        ax2.text(0.5, 0.5, 'ACF plot unavailable', ha='center', va='center')
    
    # 3. Q-Q plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('(c) Normal Q-Q Plot', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Residuals vs Fitted
    ax4 = axes[1, 1]
    ax4.scatter(fitted, residuals, alpha=0.6, s=30)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Add trend line
    z = np.polyfit(fitted, residuals, 1)
    p = np.poly1d(z)
    ax4.plot(fitted, p(fitted), "b--", linewidth=2, label='Trend')
    
    ax4.set_xlabel('Fitted Values', fontweight='medium')
    ax4.set_ylabel('Residuals', fontweight='medium')
    ax4.set_title('(d) Residuals vs Fitted', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle(f'Residual Diagnostics: {title}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig


def run_enhanced_diagnostic_tests(model, data, model_base=None):
    """
    Enhanced diagnostic tests with detailed reporting
    """
    print("\n" + "="*70)
    print("ENHANCED DIAGNOSTIC TESTS")
    print("="*70)
    
    # Use base model for diagnostics if available (for Newey-West)
    diag_model = model_base if model_base is not None else model
    residuals = diag_model.resid
    
    # 1. Durbin-Watson
    dw_stat = durbin_watson(residuals)
    print(f"\n1. Durbin-Watson Test: {dw_stat:.3f}")
    if 1.5 <= dw_stat <= 2.5:
        print(f"   ✓ No serious autocorrelation")
    elif dw_stat < 1.5:
        print(f"   ⚠ Positive autocorrelation detected")
        print(f"   → Consider: ARIMAX, Newey-West SE, or AR terms")
    else:
        print(f"   ⚠ Negative autocorrelation detected")
    
    # 2. Ljung-Box Test (more powerful than DW)
    print(f"\n2. Ljung-Box Test (autocorrelation at multiple lags):")
    try:
        lb_result = acorr_ljungbox(residuals, lags=[5, 10, 15], return_df=True)
        for lag in [5, 10, 15]:
            if lag in lb_result.index:
                pval = lb_result.loc[lag, 'lb_pvalue']
                stat = lb_result.loc[lag, 'lb_stat']
                print(f"   Lag {lag}: χ²={stat:.2f}, p={pval:.4f} {'⚠' if pval < 0.05 else '✓'}")
    except:
        print("   Could not compute")
    
    # 3. Breusch-Pagan (heteroscedasticity)
    print(f"\n3. Breusch-Pagan Test (heteroscedasticity):")
    try:
        _, bp_pvalue, _, _ = het_breuschpagan(residuals, diag_model.model.exog)
        print(f"   p-value: {bp_pvalue:.4f}")
        if bp_pvalue > 0.05:
            print(f"   ✓ Homoscedastic (constant variance)")
        else:
            print(f"   ⚠ Heteroscedastic (non-constant variance)")
            print(f"   → Consider: WLS, robust SE, or log transformation")
    except:
        print("   Could not compute")
    
    # 4. Shapiro-Wilk (normality)
    if len(residuals) < 5000:
        _, sw_pvalue = stats.shapiro(residuals)
        print(f"\n4. Shapiro-Wilk Test (normality): p={sw_pvalue:.4f}")
        print(f"   {'✓ Normal residuals' if sw_pvalue > 0.05 else '⚠ Non-normal residuals'}")
    
    # 5. Influential outliers
    try:
        influence = diag_model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / len(data)
        outliers = np.where(cooks_d > threshold)[0]
        print(f"\n5. Influential Outliers (Cook's D > {threshold:.4f}): {len(outliers)}")
        if len(outliers) > 0 and len(outliers) < 10:
            print(f"   Indices: {outliers}")
            if 'date' in data.columns:
                outlier_dates = data.iloc[outliers]['date'].dt.strftime('%Y-%m').tolist()
                print(f"   Dates: {outlier_dates}")
    except:
        print("\n5. Could not compute Cook's distance")
    
    print("="*70)


# ============================================================================
# PART 5: COMPARISON TABLE
# ============================================================================

def create_method_comparison_table(results_dict):
    """Create table comparing different estimation methods"""
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    comparison_data = []
    intervention_name = None
    
    for method, result in results_dict.items():
        model = result['model']
        intervention_name = result['intervention_name']
        post_var = f'{intervention_name}_post'
        time_var = f'{intervention_name}_time_since'
        
        try:
            if 'ARIMAX' in result['method']:
                # For ARIMAX: extract x2 and x3
                params = model.params
                pvalues = model.pvalues
                bse = model.bse
                
                # Get parameter names
                if hasattr(params, 'index'):
                    param_names = list(params.index)
                    params_dict = params.to_dict()
                    pvalues_dict = pvalues.to_dict()
                    bse_dict = bse.to_dict()
                else:
                    try:
                        param_names = list(model.param_names)
                    except:
                        param_names = [f"param_{i}" for i in range(len(params))]
                    params_dict = dict(zip(param_names, params))
                    pvalues_dict = dict(zip(param_names, pvalues))
                    bse_dict = dict(zip(param_names, bse))
                
                # Extract x2 (immediate) and x3 (trend)
                if 'x2' in params_dict and 'x3' in params_dict:
                    imm_coef = params_dict['x2']
                    imm_se = bse_dict['x2']
                    imm_pval = pvalues_dict['x2']
                    
                    trend_coef = params_dict['x3']
                    trend_se = bse_dict['x3']
                    trend_pval = pvalues_dict['x3']
                    
                    rsquared = np.nan
                else:
                    imm_coef = imm_pval = imm_se = np.nan
                    trend_coef = trend_pval = trend_se = np.nan
                    rsquared = np.nan
                    
            else:
                # For OLS, WLS, Newey-West
                params = model.params
                pvalues = model.pvalues
                bse = model.bse
                
                if hasattr(params, 'get'):
                    imm_coef = params.get(post_var, np.nan)
                    imm_pval = pvalues.get(post_var, np.nan)
                    imm_se = bse.get(post_var, np.nan)
                    
                    trend_coef = params.get(time_var, np.nan)
                    trend_pval = pvalues.get(time_var, np.nan)
                    trend_se = bse.get(time_var, np.nan)
                elif hasattr(params, '__getitem__'):
                    try:
                        imm_coef = params[post_var]
                        imm_pval = pvalues[post_var]
                        imm_se = bse[post_var]
                        
                        trend_coef = params[time_var]
                        trend_pval = pvalues[time_var]
                        trend_se = bse[time_var]
                    except:
                        imm_coef = imm_pval = imm_se = np.nan
                        trend_coef = trend_pval = trend_se = np.nan
                else:
                    imm_coef = imm_pval = imm_se = np.nan
                    trend_coef = trend_pval = trend_se = np.nan
                
                rsquared = getattr(model, 'rsquared', np.nan)
                
        except Exception as e:
            print(f"  Warning: Could not extract coefficients for {method}: {e}")
            imm_coef = imm_pval = imm_se = np.nan
            trend_coef = trend_pval = trend_se = np.nan
            rsquared = np.nan
        
        comparison_data.append({
            'Method': result['method'],
            'Immediate_Coef': imm_coef,
            'Immediate_SE': imm_se,
            'Immediate_p': imm_pval,
            'Trend_Coef': trend_coef,
            'Trend_SE': trend_se,
            'Trend_p': trend_pval,
            'R²': rsquared
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Print formatted table
    print(f"\nIntervention: {intervention_name}")
    print("-" * 70)
    
    for _, row in df_comparison.iterrows():
        print(f"\n{row['Method']}:")
        if not np.isnan(row['Immediate_Coef']):
            pct_change = (np.exp(row['Immediate_Coef']) - 1) * 100
            print(f"  Immediate Effect: {row['Immediate_Coef']:.4f} ({pct_change:+.1f}%)")
            print(f"    SE: {row['Immediate_SE']:.4f}, p: {row['Immediate_p']:.4f}", end="")
            sig = '***' if row['Immediate_p'] < 0.001 else '**' if row['Immediate_p'] < 0.01 else '*' if row['Immediate_p'] < 0.05 else 'ns'
            print(f" {sig}")
        else:
            print(f"  Immediate Effect: Could not extract")
        
        if not np.isnan(row['Trend_Coef']):
            print(f"  Trend Change: {row['Trend_Coef']:.4f}")
            print(f"    SE: {row['Trend_SE']:.4f}, p: {row['Trend_p']:.4f}", end="")
            sig = '***' if row['Trend_p'] < 0.001 else '**' if row['Trend_p'] < 0.01 else '*' if row['Trend_p'] < 0.05 else 'ns'
            print(f" {sig}")
        else:
            print(f"  Trend Change: Could not extract")
        
        if not np.isnan(row['R²']):
            print(f"  R²: {row['R²']:.4f}")
    
    print("="*70)
    
    return df_comparison

# ===========================================================================
# PART 6: PUBLICATION TABLE
# ===========================================================================

def create_publication_table(all_results):
    """Create publication-ready results table from all interventions"""
    
    table_data = []
    
    for interv_name, methods in all_results.items():
        if 'arimax' not in methods:
            continue
            
        arimax_result = methods['arimax']
        model = arimax_result['model']
        
        # Get parameters
        params = model.params
        pvalues = model.pvalues
        bse = model.bse
        
        # Handle both pandas Series and numpy array
        if hasattr(params, 'index'):
            params_dict = params.to_dict()
            pvalues_dict = pvalues.to_dict()
            bse_dict = bse.to_dict()
        else:
            try:
                param_names = list(model.param_names)
            except:
                param_names = [f"param_{i}" for i in range(len(params))]
            params_dict = dict(zip(param_names, params))
            pvalues_dict = dict(zip(param_names, pvalues))
            bse_dict = dict(zip(param_names, bse))
        
        # Extract x2 and x3
        if 'x2' in params_dict and 'x3' in params_dict:
            immediate_coef = params_dict['x2']
            immediate_se = bse_dict['x2']
            immediate_p = pvalues_dict['x2']
            
            trend_coef = params_dict['x3']
            trend_se = bse_dict['x3']
            trend_p = pvalues_dict['x3']
            
            # Calculate percentage effects
            immediate_pct = (np.exp(immediate_coef) - 1) * 100
            
            table_data.append({
                'Intervention': INTERVENTIONS[interv_name]['description'],
                'Date': INTERVENTIONS[interv_name]['date'],
                'Immediate (log)': f"{immediate_coef:.3f}",
                'Immediate SE': f"({immediate_se:.3f})",
                'Immediate %': f"{immediate_pct:+.1f}%",
                'Imm. Sig.': '***' if immediate_p < 0.001 else '**' if immediate_p < 0.01 else '*' if immediate_p < 0.05 else '',
                'Trend': f"{trend_coef:.4f}",
                'Trend SE': f"({trend_se:.4f})",
                'Trend Sig.': '***' if trend_p < 0.001 else '**' if trend_p < 0.01 else '*' if trend_p < 0.05 else '',
                'AIC': f"{model.aic:.1f}",
                'DW': f"{durbin_watson(model.resid):.2f}"
            })
    
    df_table = pd.DataFrame(table_data)
    
    print("\n" + "="*70)
    print("PUBLICATION TABLE")
    print("="*70)
    print(df_table.to_string(index=False))
    
    # Save
    df_table.to_csv('itsa_results/csv/itsa_results_table.csv', index=False)
    print("\n✓ Saved: itsa_results_table.csv")
    
    # LaTeX
    latex_table = df_table.to_latex(
        index=False,
        caption='ARIMAX-ITSA Results: Impact of Policy Interventions on EV Adoption',
        label='tab:itsa_results',
        escape=False
    )
    
    with open('./itsa_results/csv/itsa_results_table.tex', 'w') as f:
        f.write(latex_table)
    print("✓ Saved: itsa_results_table.tex")
    
    return df_table

# ============================================================================
# PART 7: MAIN PIPELINE
# ============================================================================

def main_robust_analysis_pipeline(df, outcome='EV + Hybrid', date_col='date',
                                   methods=['ols', 'newey-west', 'arimax'],
                                   arimax_order='auto'):
    """
    Complete robust ITSA pipeline comparing multiple methods
    
    Parameters:
    -----------
    methods : list of str, methods to compare
              Options: 'ols', 'newey-west', 'wls', 'arimax', 'ar-terms'
    arimax_order : tuple or 'auto', ARIMAX order. 'auto' finds best order via AIC.
    """
    print("\n" + "="*70)
    print("ROBUST INTERRUPTED TIME SERIES ANALYSIS")
    print("="*70)
    
    # Prepare data
    print("\n[1/3] Preparing data...")
    df_prepared = prepare_itsa_data(df, date_col=date_col, outcome_col=outcome)
    df_prepared = add_all_interventions(df_prepared, date_col=date_col)
    print(f"✓ Data prepared: {len(df_prepared)} observations")
    
    # Run analyses for each intervention
    print("\n[2/3] Running robust analyses...")
    all_results = {}
    
    for interv_name, interv_info in INTERVENTIONS.items():
        print(f"\n{'='*70}")
        print(f"Analyzing: {interv_info['description']}")
        print(f"{'='*70}")
        
        method_results = {}
        
        for method in methods:
            print(f"\n  → Method: {method.upper()}")
            
            try:
                result = run_robust_single_intervention_itsa(
                    df_prepared,
                    interv_name,
                    outcome=outcome,
                    log_transform=True,
                    method=method,
                    arimax_order=arimax_order
                )
                method_results[method] = result
                
                print(f"  ✓ {result['method']} completed")
                
                # Show immediate results for ARIMAX
                if method == 'arimax':
                    print(f"\n  ARIMAX Model Summary:")
                    print(f"  AIC: {result['model'].aic:.2f}")
                    print(f"  BIC: {result['model'].bic:.2f}")
                    print(f"  Log-Likelihood: {result['model'].llf:.2f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[interv_name] = method_results
        
        # Print ARIMAX coefficients properly if available
        if 'arimax' in method_results:
            print_arimax_coefficients(method_results['arimax'], interv_info['description'])
        
        # Compare methods for this intervention
        if len(method_results) > 1:
            create_method_comparison_table(method_results)
        
        # Run diagnostics on ARIMAX if available
        if 'arimax' in method_results:
            print(f"\n{'='*70}")
            print("DIAGNOSTICS (ARIMAX Method)")
            print(f"{'='*70}")
            
            arimax_result = method_results['arimax']
            
            # Check residuals
            residuals = arimax_result['model'].resid
            dw = durbin_watson(residuals)
            
            print(f"\nPost-ARIMAX Diagnostics:")
            print(f"  Durbin-Watson: {dw:.3f}")
            
            if 1.5 <= dw <= 2.5:
                print(f"  ✓ Autocorrelation resolved!")
            else:
                print(f"  ⚠ Some autocorrelation remains (may need higher order)")
            
            # Ljung-Box test
            try:
                lb_result = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
                print(f"\n  Ljung-Box Test:")
                for lag in [5, 10]:
                    if lag in lb_result.index:
                        pval = lb_result.loc[lag, 'lb_pvalue']
                        print(f"    Lag {lag}: p={pval:.4f} {'✓' if pval > 0.05 else '⚠'}")
            except:
                pass
            
            # Diagnostic plots
            plot_residual_diagnostics(
                arimax_result['model'],
                arimax_result['data'],
                title=f"{interv_info['description']} (ARIMAX)",
                save_path=f'itsa_results/images/diagnostics_arimax_{interv_name}.png'
            )
    
    # Summary recommendations
    print("\n[3/3] Analysis complete!")
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR REPORTING")
    print("="*70)
    
    print("\n✓ PRIMARY MODEL: ARIMAX")
    print("  - Explicitly accounts for autocorrelation")
    print("  - More reliable p-values and confidence intervals")
    print("  - Check Durbin-Watson improved (should be 1.5-2.5)")
    
    print("\n✓ SENSITIVITY CHECK: Report OLS + Newey-West")
    print("  - Shows robustness of findings across methods")
    print("  - Newey-West provides conservative standard errors")
    
    print("\n✓ INTERPRETATION:")
    print("  - Focus on coefficient signs and relative magnitudes")
    print("  - If ARIMAX and Newey-West agree on significance, result is robust")
    print("  - If they disagree, report as 'suggestive but not definitive'")
    
    print("\n✓ KEY METRICS TO REPORT:")
    print("  - Immediate effect (level change)")
    print("  - Trend change (slope difference)")
    print("  - Cumulative impact")
    print("  - Model fit (AIC/BIC for ARIMAX, R² for OLS)")
    
    print("="*70)
    
    return all_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
Enhanced ITSA Script with ARIMAX - Ready to Use

Quick Start:
-----------
# 1. Load data
df = pd.read_csv('ev_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 2. Run ARIMAX-focused analysis
results = main_robust_analysis_pipeline(
    df, 
    outcome='EV + Hybrid',
    methods=['ols', 'newey-west', 'arimax'],
    arimax_order='auto'  # Automatically finds best ARIMA order
)

# 3. Access ARIMAX results
fame_arimax = results['FAME_II']['arimax']
print(fame_arimax['model'].summary())

# 4. Compare across methods
for method in ['ols', 'newey-west', 'arimax']:
    result = results['FAME_II'][method]
    print(f"{method}: {result['method']}")

Key ARIMAX Features:
-------------------
✓ Automatically searches for best ARIMA order (p,d,q)
✓ Explicitly models autocorrelation structure
✓ Provides proper counterfactual predictions
✓ More reliable inference than standard OLS
✓ Compares with Newey-West for robustness check

What ARIMAX Does:
----------------
- AR(p): Uses p past values to predict current value
- MA(q): Uses q past forecast errors
- With your interventions as exogenous variables

The script will:
1. Try multiple ARIMA orders and select best by AIC
2. Report if autocorrelation is resolved (DW statistic)
3. Show coefficient comparison across all methods
4. Generate diagnostic plots for residual checking

Expected Output:
---------------
- "Durbin-Watson: 1.8-2.2" means autocorrelation is resolved
- Ljung-Box p>0.05 confirms no remaining autocorrelation
- Coefficients similar across methods = robust findings
- Significant effects in both ARIMAX and Newey-West = strong evidence

Troubleshooting:
---------------
If ARIMAX fails to converge:
- Try fixed order: arimax_order=(1,0,0) or (2,0,1)
- Reduce max_p and max_q in find_best_arimax_order()
- Check for extreme outliers in data

If DW still low after ARIMAX:
- Try higher AR order: arimax_order=(2,0,0) or (3,0,0)
- Consider seasonal ARIMAX if monthly patterns exist
- May need structural breaks or additional control variables
    """)