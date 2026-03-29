import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick

# ==========================================
# 1. DATA EXPLORATION (EDA)
# ==========================================
def eda(df):
    # 1. Collect basic metrics
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isna().sum(),
        '% Missing': (df.isna().mean() * 100).round(2),
        'Unique Count': df.nunique()
    })
    
    # 2. Add sample values (so you can see what the data looks like)
    # We take the first 10 unique values for each column
    summary['Samples (First 10 Unique)'] = [df[col].unique()[:10] for col in df.columns]
    
    # 3. Sort by Missing Values (optional, helps find dirty data)
    summary = summary.sort_values(by='Missing Values', ascending=False)
    print(f"Dataset Shape: {df.shape}")
    print("-" * 30)
    return summary

# ==========================================
# 2. TARGET CREATION
# ==========================================
def target(selection_df, gifts_df, target_start, target_end, target_campaign_id, min_amount=30):
    """
    Creates the Target of each of the training, validation, and test sets.
    It is based on the aggregated donations of the donor in the DV time window
    and directed towards the specific campaign.
    Takes date strings in the 'yyyy-mm-dd' format.
    """
    population = selection_df.copy()
    
    # Convert string arguments to date objects
    target_start_obj = pd.to_datetime(target_start).date()
    target_end_obj   = pd.to_datetime(target_end).date()
    
    print(f"--- Processing Population (Campaign {target_campaign_id}) ---")
    print(f"   > Target Window:  {target_start_obj} to {target_end_obj}")
    
    # Strictly within the target time window
    # Strictly matching the target campaign ID
    
    # We create a boolean mask for the date first to reduce the size
    date_mask = (gifts_df['date'] >= target_start_obj) & (gifts_df['date'] <= target_end_obj)
    campaign_mask = (gifts_df['campaignID'] == target_campaign_id)
    
    # Apply combined mask
    gifts_in_window = gifts_df[date_mask & campaign_mask].copy()
    
    # Aggregate by DonorID to get total donation amount
    donor_sums = gifts_in_window.groupby('donorID')['amount'].sum().reset_index()

    # Identify donors who gave >= min_amount (30 euros) total
    target_donors = donor_sums[donor_sums['amount'] >= min_amount]['donorID'].unique()

    # 1 if they are in the 'target_donors' list, 0 otherwise
    population['target'] = population['donorID'].isin(target_donors).astype(int)
    
    # Check Statistics
    count = population['target'].sum()
    rate = population['target'].mean() * 100
    print(f"   > Total Responders (Cumulative >= €{min_amount}): {count}")
    print(f"   > Response Rate: {rate:.2f}%\n")
    
    return population

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def features(population_df, gifts_df, donors_df, campaigns_df, iv_start_date, iv_end_date):
    """
    Enriches the population DataFrame with features calculated strictly within the IV window.
    
    Parameters:
    - population_df: DataFrame containing the 'donorID' column.
    - gifts_df: DataFrame containing gift history.
    - donors_df: DataFrame containing static donor info.
    - campaigns_df: DataFrame containing campaign info.
    - iv_start_date: Start date of the Independent Variable window (string 'YYYY-MM-DD').
    - iv_end_date: End date of the Independent Variable window (string 'YYYY-MM-DD').
    
    Returns:
    - df: A DataFrame with new feature columns.
    """
    print(f"--- Adding Features ---")
    print(f"   > Observation Window: {iv_start_date} to {iv_end_date}")
    
    # 1. Initialize output dataframe
    df = population_df.copy()
    
    # Convert date strings to objects for comparison
    iv_start = pd.to_datetime(iv_start_date).date()
    iv_end   = pd.to_datetime(iv_end_date).date()
    
    # ---------------------------------------------------------
    # 2. MERGE STATIC FEATURES
    # ---------------------------------------------------------
    # Merge relevant columns from the Donors table
    df = df.merge(donors_df[['donorID', 'gender', 'zipcode', 'province', 'region', 'language', 'dateOfBirth', 'isGenderMissing']],
                on='donorID',
                how='inner')
    
    # ---------------------------------------------------------
    # 3. FILTER HISTORY (IV Window)
    # ---------------------------------------------------------
    # Create a subset of gifts that happened strictly within the observation window
    gifts_iv = gifts_df[
        (gifts_df['date'] >= iv_start) & 
        (gifts_df['date'] <= iv_end)
    ].copy()

    # ---------------------------------------------------------
    # 4. AGE & DEMOGRAPHICS
    # ---------------------------------------------------------
    # Calculate Age at the end of the observation window
    # We enforce datetime format to ensure subtraction works
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'])
    df['age'] = round((pd.to_datetime(iv_end) - df['dateOfBirth']).dt.days / 365.25, 2)
    
    # Bin Age into groups
    df['age_group'] = "Unknown" 
    # Youth: Anything below 30 (e.g., 29.99)
    df.loc[df['age'] < 30.0, "age_group"] = "Youth"
    # Adult: 30 up to (but not including) 60
    df.loc[(df['age'] >= 30.0) & (df['age'] < 60.0), "age_group"] = "Adult"
    # Senior: 60 up to (but not including) 66
    df.loc[(df['age'] >= 60.0) & (df['age'] < 66.0), "age_group"] = "Senior"
    # Retired: 66 and older
    df.loc[df['age'] >= 66.0, "age_group"] = "Retired"

    # ---------------------------------------------------------
    # 5. FREQUENCY (Count of Gifts)
    # ---------------------------------------------------------    
    freq_df = gifts_iv.groupby('donorID').size().reset_index(name='frequency')
    df = df.merge(freq_df, on='donorID', how='left')
    
    # Fill missing values with 0 (implies no gifts in window)
    df['frequency'] = df['frequency'].fillna(0).astype(int)

    # ---------------------------------------------------------
    # 6. CAMPAIGN BEHAVIOR (Spontaneous vs. Solicited)
    # ---------------------------------------------------------
    # Identify spontaneous gifts (CampaignID == '-1')
    # Using robust string conversion to handle potential float/string mismatches
    gifts_iv['is_spontaneous'] = gifts_iv['campaignID'].astype(str) == '-1'
    
    # Check if donor has EVER given spontaneously
    indep_flag = gifts_iv.groupby('donorID')['is_spontaneous'].max().reset_index(name='is_independent_donor')
    df = df.merge(indep_flag, on='donorID', how='left')
    
    # Fill missing values with 0
    df['is_independent_donor'] = df['is_independent_donor'].fillna(0).astype(int)

    # ---------------------------------------------------------
    # 6.1 CAMPAIGN BEHAVIOR (Frequency of independent donations)
    # ---------------------------------------------------------
    # Number of times the donor has been an independent donor
    num_independent_donations = gifts_iv.groupby("donorID")["campaignID"].apply(lambda x: (x == "-1").sum()).reset_index(name="frequency_independent")
    num_independent_donations[num_independent_donations["frequency_independent"] > 0]
    df = df.merge(num_independent_donations, on='donorID', how='left')
    df['frequency_independent'] = df['frequency_independent'].fillna(0)

    # ---------------------------------------------------------
    # 7. RECENCY (Days since last gift)
    # ---------------------------------------------------------
    # Find the latest gift date per donor
    last_gift = gifts_iv.groupby('donorID')['date'].max().reset_index()
    last_gift.columns = ['donorID', 'last_gift_date']
    df = df.merge(last_gift, on='donorID', how='left')
    
    # Calculate days elapsed
    df['recency'] = (pd.to_datetime(iv_end) - pd.to_datetime(df['last_gift_date'])).dt.days
    
    # Fill missing recency with a high number (approx 5 years and 1 day)
    df['recency'] = df['recency'].fillna(1826).astype(int)
    df = df.drop(columns='last_gift_date')

    # ---------------------------------------------------------
    # 8. SENIORITY (Years as Donor)
    # ---------------------------------------------------------
    # Find the first gift date per donor
    first_gift = gifts_iv.groupby('donorID')['date'].min().reset_index()
    first_gift.columns = ['donorID', 'first_gift_date']
    df = df.merge(first_gift, on='donorID', how='left')
    
    # Calculate years elapsed
    df['years_as_donor'] = (pd.to_datetime(iv_end) - pd.to_datetime(df['first_gift_date'])).dt.days / 365.25
    
    # Fill missing seniority with 0
    df['years_as_donor'] = df['years_as_donor'].fillna(0)
    df = df.drop(columns='first_gift_date')

    # ---------------------------------------------------------
    # 9. MONETARY STATISTICS (Total, Avg, Min, Max, Std)
    # ---------------------------------------------------------
    monetary_stats = gifts_iv.groupby('donorID')['amount'].agg(['sum', 'mean', 'max', 'min', 'std']).reset_index()
    monetary_stats.columns = ['donorID', 'total_amount', 'avg_amount', 'max_amount', 'min_amount', 'volatility']
    df = df.merge(monetary_stats, on='donorID', how='left')

    # Fill NaNs with 0 (for donors with no history)
    cols_to_fill = ['total_amount', 'avg_amount', 'max_amount', 'min_amount', 'volatility']
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    # ---------------------------------------------------------
    # NEW: Binning Monetary Features
    # ---------------------------------------------------------
    # We use qcut to create bins with equal number of donors (Quantiles)
    # This handles skewed data better than fixed numbers.
    
    # 1. Bin Average Amount (5 Groups: Very Low to Very High)
    # labels=False returns integers 0-4, which is easier to work with
    df['avg_amount_bin'] = pd.qcut(df['avg_amount'].rank(method='first'), q=5, labels=False)
    
    # 2. Bin Total Amount
    df['total_amount_bin'] = pd.qcut(df['total_amount'].rank(method='first'), q=5, labels=False)
    
    # OPTIONAL: Manually defined bins (Business Logic)
    # Useful if you want clear categories like "Small Donor" vs "Major Donor"
    # Bins: 0-5, 5-15, 15-30, 30-60, 60+
    bins_custom = [-1, 5, 15, 30, 60, 999999]
    labels_custom = ['Micro', 'Small', 'Medium', 'Large', 'Major']
    df['avg_amount_cat'] = pd.cut(df['avg_amount'], bins=bins_custom, labels=labels_custom)

    # ---------------------------------------------------------
    # 10. LAST GIFT AMOUNT
    # ---------------------------------------------------------
    # 1. Aggregate gifts by Donor AND Date first
    #    This sums multiple gifts on the same day (e.g., 3x €3 becomes €9)
    daily_gifts = gifts_iv.groupby(['donorID', 'date'])['amount'].sum().reset_index()
    
    # 2. Sort by date to find the very last day they donated
    #    We sort ascending, so tail(1) is the latest date
    last_gift_df = daily_gifts.sort_values('date').groupby('donorID').tail(1)[['donorID', 'amount']]
    
    # 3. Rename and Merge
    last_gift_df.columns = ['donorID', 'last_gift_amount']
    
    df = df.merge(last_gift_df, on='donorID', how='left')
    df['last_gift_amount'] = df['last_gift_amount'].fillna(0)

    # ---------------------------------------------------------
    # 11. DONOR SEGMENTATION (Top % Flags)
    # ---------------------------------------------------------
    # Calculate 80th and 90th percentile thresholds based on Total Amount
    threshold_top_10 = df['total_amount'].quantile(0.90)
    threshold_top_20 = df['total_amount'].quantile(0.80)
    
    # Create binary flags
    df['is_top_10_percent'] = (df['total_amount'] >= threshold_top_10).astype(int)
    df['is_top_20_percent'] = (df['total_amount'] >= threshold_top_20).astype(int)

    # ---------------------------------------------------------
    # 12. SEASONALITY ALIGNMENT (NEW FEATURE)
    # ---------------------------------------------------------
    # Define Campaign Window based on the IV End Date + Gap
    # Gap is approx 7 days. Campaign duration approx 30-44 days.
    camp_start_date = pd.to_datetime(iv_end) + pd.Timedelta(days=7)
    camp_end_date   = camp_start_date + pd.Timedelta(days=44) # Using your logic of 44 days
    
    start_doy = camp_start_date.dayofyear
    end_doy   = camp_end_date.dayofyear
    
    # Calculate Day of Year for all past gifts
    gifts_iv['doy'] = pd.to_datetime(gifts_iv['date']).dt.dayofyear
    
    # Check if past gifts fall in this window
    if start_doy <= end_doy:
        # Standard case (e.g., June 1 to June 30)
        gifts_iv['in_season'] = (gifts_iv['doy'] >= start_doy) & (gifts_iv['doy'] <= end_doy)
    else:
        # Wrap-around case (e.g., Dec 20 to Jan 20)
        gifts_iv['in_season'] = (gifts_iv['doy'] >= start_doy) | (gifts_iv['doy'] <= end_doy)
        
    # Aggregate count of seasonal gifts per donor
    season_counts = gifts_iv[gifts_iv['in_season']].groupby('donorID').size().reset_index(name='donations_in_campaign_period')
    
    # Merge and Fill
    df = df.merge(season_counts, on='donorID', how='left')
    df['donations_in_campaign_period'] = df['donations_in_campaign_period'].fillna(0).astype(int)

    # # ---------------------------------------------------------
    # # NEW: ADVANCED FEATURE ENGINEERING
    # # ---------------------------------------------------------
    
    # # 1. LIFECYCLE STAGE (Recency Binning)
    # # Logic: Segment donors by how long they've been "cold"
    # # Bins: 0-12mo (Active), 12-24mo (Lapsing), 24-60mo (Inactive), 60mo+ (Lost)
    # bins_rec = [-1, 365, 730, 1825, 99999]
    # labels_rec = ['Active', 'Lapsing', 'Inactive', 'Lost']
    # df['lifecycle_stage'] = pd.cut(df['recency'], bins=bins_rec, labels=labels_rec)

    # # 2. FREQUENCY STATUS (Loyalty Binning)
    # # Logic: Segment donors by habit
    # # Bins: 0 (Non), 1 (One-time), 2-5 (Repeat), 5+ (Loyal)
    # bins_freq = [-1, 0, 1, 5, 9999]
    # labels_freq = ['NonDonor', 'OneTime', 'Repeat', 'Loyal']
    # df['frequency_status'] = pd.cut(df['frequency'], bins=bins_freq, labels=labels_freq)

    # # 3. CONSISTENCY SCORE (Ratio)
    # # Logic: Min/Max. Close to 1 = Consistent amounts. Close to 0 = Sporadic amounts.
    # # We use np.where to handle the case where max_amount is 0 (avoid division by zero)
    # df['consistency_score'] = np.where(df['max_amount'] > 0, 
    #                                 df['min_amount'] / df['max_amount'], 
    #                                 0)

    # # 4. ANNUAL VALUE (Velocity)
    # # Logic: Total Amount / Years active. This gives "Value per Year".
    # # We add a tiny epsilon (0.1) to years_as_donor to avoid dividing by zero for new prospects
    # df['annual_value'] = df['total_amount'] / (df['years_as_donor'] + 0.1)

    # ---------------------------------------------------------
    # 13. CLEANUP
    # ---------------------------------------------------------
    # Drop temporary or unused columns
    drop_cols = ['feature_date', 'zipcode', 'dateOfBirth']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    print("   > Done.")
    return df

# ==========================================
# 4. PREPROCESSING (Dummy & Scaling)
# ==========================================
def dummy(df, categorical_cols=None):
    """
    Prepares the table for machine learning by:
    One-Hot Encoding categorical variables (creating dummy variables).
    """
    df_final = df.copy()
    
    # Define default categorical columns if none provided
    categorical_cols = ['gender', 'language', 'province', 'region', 'age_group', 'avg_amount_bin', 'total_amount_bin', 'avg_amount_cat']
    
    print(f"--- Finalizing Table ---")
    print(f"   > Input shape: {df_final.shape}")
    print(f"   > Encoding columns: {categorical_cols}")
    
    # Create Dummy Variables
    # drop_first=True avoids multicollinearity (e.g., if you have Is_Male, you don't need Is_Female)
    df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)
    
    print(f"   > Output shape: {df_final.shape}")
    print("   > Done.")
    
    return df_final

def align_datasets(train_df, test_df, score_df):
    """
    Ensures Test and Score datasets have exactly the same columns 
    in the same order as the Training dataset.
    
    Logic:
    1. If a column is in Train but missing in Test/Score -> Add it (fill with 0).
    2. If a column is in Test/Score but missing in Train -> Drop it (model doesn't know it).
    3. Reorder Test/Score columns to match Train exactly.
    """
    print("--- Aligning Datasets ---")
    
    # 1. Get the "Source of Truth" columns (from Train)
    # We maintain the exact order
    train_cols = train_df.columns.tolist()
    
    print(f"   > Reference columns (Train): {len(train_cols)}")
    
    # 2. Align Test Set
    # reindex handles both adding missing cols (fill_value=0) and dropping extras
    test_aligned = test_df.reindex(columns=train_cols, fill_value=0)
    
    # 3. Align Score Set
    score_aligned = score_df.reindex(columns=train_cols, fill_value=0)
    
    # 4. Final Safety Check
    if list(train_df.columns) == list(test_aligned.columns) == list(score_aligned.columns):
        print("   > SUCCESS: All column lists match exactly.")
    else:
        print("   > WARNING: Mismatch detected even after alignment!")
        
    print(f"   > Final Shapes: Train {train_df.shape}, Test {test_aligned.shape}, Score {score_aligned.shape}")
    
    return train_df, test_aligned, score_aligned

def scale_features(df, scaler=None):
    """
    Standardizes ALL numeric columns that are NOT binary (0/1).
    
    Parameters:
    - df: The DataFrame to scale.
    - scaler: An existing StandardScaler object (optional). 
    
    Returns:
    - df_scaled: The DataFrame with scaled columns.
    - scaler: The scaler object used.
    """
    df_scaled = df.copy()
    
    # 1. Identify Numeric Columns
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    # 2. Filter out Binary Columns (Target, Gender flags, etc.)
    # Logic: If a column has more than 2 unique values, OR if the values are not just {0, 1}, we scale it.
    cols_to_scale = []
    
    for col in numeric_cols:
        # Skip 'target' and 'donorID' explicitly just in case
        if col in ['target', 'donorID']:
            continue
            
        unique_vals = df_scaled[col].dropna().unique()
        
        # Check if it looks binary (only contains 0 and 1)
        is_binary = set(unique_vals).issubset({0, 1})
        
        if not is_binary:
            cols_to_scale.append(col)
            
    if not cols_to_scale:
        print("Warning: No non-binary numeric columns found to scale.")
        return df_scaled, scaler

    print(f"Scaling {len(cols_to_scale)} non-binary columns: {cols_to_scale}")

    # 3. Apply Scaling
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        print("   > Created and fitted a NEW Scaler.")
    else:
        # Safety check: Ensure the test set has the exact same columns
        # If test set is missing a column found in train, this might crash, 
        # but your align_datasets function handles that!
        df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])
        print("   > Used EXISTING Scaler parameters.")
        
    return df_scaled, scaler

# ==========================================
# 5. ANALYSIS & VISUALIZATION
# ==========================================
def corr(df, threshold=0.90):
    """
    Calculates and plots the correlation matrix for numeric columns.
    Prints a list of highly correlated features (above the threshold).
    Does NOT drop any columns.
    """
    # 1. Exclude 'donorID' from the visualization (it's not a predictive feature)
    df_numeric = df.drop(columns=['donorID'])
        
    print(f"Calculating correlation on {df_numeric.shape[1]} numeric columns...")

    # 2. Compute Correlation Matrix (Absolute values)
    corr_matrix = df_numeric.corr().abs()

    # 3. Plot Heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix, 
        cmap='coolwarm', 
        annot=True,            # Show numbers
        fmt=".2f",             # 2 decimal places
        annot_kws={"size": 8}, # Small font to fit many features
        linewidths=0.5
    )
    plt.title("Correlation Matrix")
    plt.show()

    # 4. Identify High Correlations (The "Redundant" Features)
    # We look at the upper triangle of the matrix to avoid duplicates (A-B is same as B-A)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns where correlation > threshold
    high_corr_features = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"\n--- Analysis Results ---")
    if len(high_corr_features) > 0:
        print(f"Found {len(high_corr_features)} features with correlation > {threshold}:")
        print(high_corr_features)
        print("\nTip: You should consider dropping these to reduce multicollinearity.")
    else:
        print(f"No features found with correlation > {threshold}.")
        
    return high_corr_features

# ==========================================
# 6. CUMULATIVE GAINS & RESPONSE RATE
# ==========================================
def plot_model_performance(y_true, y_probs, target_baseline=1.8):
    """
    Plots Cumulative Gains, Standardized Response Rate, and Lift.
    - Graph: Shows points every 10% (Deciles) to keep it clean.
    - Summary Table: Shows points every 5% (Vingtiles) for more detail.
    """
    # 1. Prepare Data
    data = pd.DataFrame({'y_true': y_true, 'prob': y_probs})
    data = data.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    total_donors = data['y_true'].sum()
    n_rows = len(data)
    
    data['cum_rows'] = np.arange(1, n_rows + 1)
    data['cum_donors'] = data['y_true'].cumsum()
    
    # 2. Metrics
    data['percent_contacted'] = data['cum_rows'] / n_rows
    data['percent_captured'] = data['cum_donors'] / total_donors
    
    # -- A. Standardized Response Rate --
    data['raw_rate'] = data['cum_donors'] / data['cum_rows']
    actual_mean = data['y_true'].mean()
    target_mean = target_baseline / 100.0
    scaling_factor = target_mean / actual_mean
    data['scaled_rate'] = data['raw_rate'] * scaling_factor
    
    # -- B. Lift Calculation --
    data['lift'] = data['raw_rate'] / actual_mean

    # 3. Define Marker Points
    # A. For the PLOT (Every 10% - Cleaner)
    indices_plot = [int(n_rows * i / 10) - 1 for i in range(1, 11)]
    indices_plot[-1] = n_rows - 1

    # B. For the TABLE (Every 5% - More Detail)
    indices_table = [int(n_rows * i / 20) - 1 for i in range(1, 21)]
    indices_table[-1] = n_rows - 1 

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # === CHART A: Cumulative Gains ===
    ax1 = axes[0]
    ax1.plot(data['percent_contacted'], data['percent_captured'], label='CatBoost', 
            linewidth=2, color='tab:blue', marker='o', markersize=6, markevery=indices_plot)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_title('Cumulative Gains (Capture Rate)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('% Database Contacted')
    ax1.set_ylabel('% Total Donors Captured')
    
    # Annotations for Gains (Only at Plot indices)
    for i in indices_plot:
        x = data.loc[i, 'percent_contacted']
        y = data.loc[i, 'percent_captured']
        ax1.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='tab:blue', fontweight='bold')

    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === CHART B: Standardized Response Rate ===
    ax2 = axes[1]
    ax2.plot(data['percent_contacted'], data['scaled_rate'], label='Model Response Rate', 
            linewidth=2, color='tab:green', marker='s', markersize=6, markevery=indices_plot)
    ax2.axhline(y=target_mean, color='k', linestyle='--', label=f'Random ({target_baseline}%)')
    
    ax2.set_title(f'Cumulative Response Rate (Baseline = {target_baseline}%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('% Database Contacted')
    ax2.set_ylabel('Response Rate (%)')
    
    # Annotations for Response Rate
    for i in indices_plot:
        x = data.loc[i, 'percent_contacted']
        y = data.loc[i, 'scaled_rate']
        if i == indices_plot[0] and y > (data['scaled_rate'].head(int(n_rows*0.05)).mean() * 1.5): continue 
        ax2.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='tab:green', fontweight='bold')

    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    top_rate = data['scaled_rate'].head(int(n_rows * 0.05)).mean()
    ax2.set_ylim(0, top_rate * 1.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === CHART C: Lift Curve ===
    ax3 = axes[2]
    ax3.plot(data['percent_contacted'], data['lift'], label='Model Lift', 
            linewidth=2, color='tab:red', marker='^', markersize=6, markevery=indices_plot)
    ax3.axhline(y=1.0, color='k', linestyle='--', label='Random (1.0)')
    
    ax3.set_title('Cumulative Lift', fontsize=14, fontweight='bold')
    ax3.set_xlabel('% Database Contacted')
    ax3.set_ylabel('Lift (Multiples of Random)')
    
    # Annotations for Lift
    for i in indices_plot:
        x = data.loc[i, 'percent_contacted']
        y = data.loc[i, 'lift']
        if i == indices_plot[0] and y > (data['lift'].head(int(n_rows*0.05)).mean() * 1.5): continue 
        ax3.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='tab:red', fontweight='bold')

    ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    top_lift = data['lift'].head(int(n_rows * 0.05)).mean()
    ax3.set_ylim(0, top_lift * 1.3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- PRINT SUMMARY TABLE (More Points Here!) ---
    print("\n" + "="*60)
    print("      DETAILED PERFORMANCE SUMMARY (5% Intervals)")
    print("="*60)
    summary = data.loc[indices_table, ['percent_contacted', 'percent_captured', 'scaled_rate', 'lift']].copy()
    summary.columns = ['% Contacted', '% Captured', 'Response Rate', 'Lift']
    
    # Formatting
    summary['% Contacted'] = summary['% Contacted'].map('{:.0%}'.format)
    summary['% Captured'] = summary['% Captured'].map('{:.1%}'.format)
    summary['Response Rate'] = summary['Response Rate'].map('{:.2%}'.format)
    summary['Lift'] = summary['Lift'].map('{:.2f}'.format)
    
    print(summary.to_string(index=False))
    print("="*60 + "\n")

# ==========================================
# 7. CUMULATIVE GAINS & RESPONSE RATE
# ==========================================
def summarize_features(df, max_categories=20):
    """
    Prints useful information about every column in the dataframe:
    - dtype
    - number of missing values
    - number of unique values
    - value counts (for categorical columns)
    - summary statistics (for numerical columns)
    """

    for col in df.columns:
        print("="*80)
        print(f"📌 Column: {col}")
        print(f"Type: {df[col].dtype}")

        # Missing and unique values
        print(f"Missing values: {df[col].isna().sum()}")
        print(f"Unique values: {df[col].nunique()}")

        # Numerical columns → summary stats
        if pd.api.types.is_numeric_dtype(df[col]):
            print("\nSummary statistics:")
            print(df[col].describe())  # count, mean, std, min, max, quartiles

        # Categorical or low-cardinality columns → value counts
        elif df[col].nunique() <= max_categories:
            print("\nValue counts:")
            print(df[col].value_counts(dropna=False))

        # High-cardinality text columns → show example values
        else:
            print("\nHigh-cardinality column — showing 10 sample values:")
            print(df[col].dropna().unique()[:10])

        print("\n")  # spacing for readability