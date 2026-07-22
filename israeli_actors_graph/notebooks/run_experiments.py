"""
Feature Engineering Experiments
================================
Testing multiple feature combinations to improve link prediction model.

Based on: Course Notebook 11 (Link Prediction and Edge Classification)
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

sys.path.append('../src')
import link_prediction as lp
import link_features as lf
import graph_build as gb

np.random.seed(42)

print("="*80)
print(" " * 20 + "FEATURE ENGINEERING EXPERIMENTS")
print("="*80)

# =============================================================================
# 1. Load Data
# =============================================================================
print("\n[1/11] Loading data...")
cast_edges = pd.read_csv('../data/processed/cast_edges.csv')
print(f"  ✓ Cast edges: {len(cast_edges):,}")
print(f"  ✓ Actors: {cast_edges['actor_slug'].nunique():,}")
print(f"  ✓ Movies: {cast_edges['movie_slug'].nunique():,}")

# =============================================================================
# 2. Build Temporal Graphs
# =============================================================================
print("\n[2/11] Building temporal graphs...")
G_1990_2015 = lp.build_graph_for_years(cast_edges, 1990, 2015)
G_1990_2020 = lp.build_graph_for_years(cast_edges, 1990, 2020)
G_1990_2025 = lp.build_graph_for_years(cast_edges, 1990, 2025)
print(f"  ✓ G_1990_2015: {G_1990_2015.number_of_nodes()} nodes, {G_1990_2015.number_of_edges()} edges")
print(f"  ✓ G_1990_2020: {G_1990_2020.number_of_nodes()} nodes, {G_1990_2020.number_of_edges()} edges")

# =============================================================================
# 3. Extract Train/Test Pairs
# =============================================================================
print("\n[3/11] Extracting train/test pairs...")
train_pos = lp.get_new_pairs(G_1990_2015, G_1990_2020)
train_neg = lp.sample_negatives(G_1990_2015, len(train_pos), exclude_pairs=set(train_pos), seed=42)
test_pos = lp.get_new_pairs(G_1990_2020, G_1990_2025)
test_neg = lp.sample_negatives(G_1990_2020, len(test_pos), exclude_pairs=set(test_pos), seed=42)

train_pairs = train_pos + train_neg
test_pairs = test_pos + test_neg
y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

print(f"  ✓ Train: {len(train_pos)} pos, {len(train_neg)} neg")
print(f"  ✓ Test:  {len(test_pos)} pos, {len(test_neg)} neg")

# =============================================================================
# 4. Feature Engineering Functions
# =============================================================================
def add_resource_allocation(G, pairs):
    """Add Resource Allocation Index"""
    ra_map = {}
    for u, v, score in nx.resource_allocation_index(G, pairs):
        ra_map[(u, v)] = score
    return np.array([ra_map.get(p, 0.0) for p in pairs]).reshape(-1, 1)


def add_node_features(G, pairs):
    """Add triangles, clustering, pagerank"""
    triangles = nx.triangles(G)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, max_iter=100)
    
    features = []
    for u, v in pairs:
        features.append([
            triangles.get(u, 0),
            triangles.get(v, 0),
            np.sqrt(triangles.get(u, 0) * triangles.get(v, 0)),
            clustering.get(u, 0.0),
            clustering.get(v, 0.0),
            (clustering.get(u, 0.0) + clustering.get(v, 0.0)) / 2,
            pagerank.get(u, 0.0),
            pagerank.get(v, 0.0),
            pagerank.get(u, 0.0) * pagerank.get(v, 0.0),
        ])
    return np.array(features)


def add_temporal_features(G, pairs, cast_edges, current_year):
    """Add career overlap, collaboration frequency"""
    actor_years = cast_edges.groupby('actor_slug')['year'].apply(set).to_dict()
    
    features = []
    for u, v in pairs:
        u_years = actor_years.get(u, set())
        v_years = actor_years.get(v, set())
        
        overlap = len(u_years & v_years) if u_years and v_years else 0
        u_career = max(u_years) - min(u_years) + 1 if u_years else 0
        v_career = max(v_years) - min(v_years) + 1 if v_years else 0
        u_recent = 1 if u_years and max(u_years) >= current_year - 3 else 0
        v_recent = 1 if v_years and max(v_years) >= current_year - 3 else 0
        collab_count = G.number_of_edges(u, v) if G.has_edge(u, v) else 0
        
        features.append([
            overlap, u_career, v_career, u_recent, v_recent, 
            collab_count, len(u_years), len(v_years)
        ])
    return np.array(features)

# =============================================================================
# 5. Experiment 1: BASELINE
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: BASELINE (24 features)")
print("="*80)

print("[4/11] Computing baseline features...")
centralities_train = lp.compute_centralities(G_1990_2015)
svd_emb_train = lf.svd_node_embeddings(G_1990_2015, n_components=32, seed=42)
X_train_base, _ = lf.build_feature_matrix(
    G_1990_2015, train_pairs, centralities_train, svd_emb_train, 
    n2v_emb=None, precomputed_sp=None, years_ahead=5
)

centralities_test = lp.compute_centralities(G_1990_2020)
svd_emb_test = lf.svd_node_embeddings(G_1990_2020, n_components=32, seed=42)
X_test_base, _ = lf.build_feature_matrix(
    G_1990_2020, test_pairs, centralities_test, svd_emb_test,
    n2v_emb=None, precomputed_sp=None, years_ahead=5
)

print("[5/11] Training baseline models...")
lr_base, scaler_base = lp.train_logistic_regression(X_train_base, y_train, seed=42)
rf_base = lp.train_random_forest(X_train_base, y_train, seed=42)

metrics_lr_base = lp.evaluate_model(lr_base, X_test_base, y_test, scaler_base)
metrics_rf_base = lp.evaluate_model(rf_base, X_test_base, y_test, None)

print(f"\n  LR  - AUC: {metrics_lr_base['auc_roc']:.4f}")
print(f"  RF  - AUC: {metrics_rf_base['auc_roc']:.4f}")

# =============================================================================
# 6. Experiment 2: +Resource Allocation
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: +Resource Allocation (25 features)")
print("="*80)

print("[6/11] Adding Resource Allocation...")
ra_train = add_resource_allocation(G_1990_2015, train_pairs)
ra_test = add_resource_allocation(G_1990_2020, test_pairs)
X_train_ra = np.column_stack([X_train_base, ra_train])
X_test_ra = np.column_stack([X_test_base, ra_test])

lr_ra, scaler_ra = lp.train_logistic_regression(X_train_ra, y_train, seed=42)
rf_ra = lp.train_random_forest(X_train_ra, y_train, seed=42)

metrics_lr_ra = lp.evaluate_model(lr_ra, X_test_ra, y_test, scaler_ra)
metrics_rf_ra = lp.evaluate_model(rf_ra, X_test_ra, y_test, None)

print(f"\n  LR  - AUC: {metrics_lr_ra['auc_roc']:.4f} (Δ={metrics_lr_ra['auc_roc']-metrics_lr_base['auc_roc']:+.4f})")
print(f"  RF  - AUC: {metrics_rf_ra['auc_roc']:.4f} (Δ={metrics_rf_ra['auc_roc']-metrics_rf_base['auc_roc']:+.4f})")

# =============================================================================
# 7. Experiment 3: +Node Features
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: +Node Features (33 features)")
print("="*80)

print("[7/11] Adding node features...")
node_feat_train = add_node_features(G_1990_2015, train_pairs)
node_feat_test = add_node_features(G_1990_2020, test_pairs)
X_train_node = np.column_stack([X_train_base, node_feat_train])
X_test_node = np.column_stack([X_test_base, node_feat_test])

lr_node, scaler_node = lp.train_logistic_regression(X_train_node, y_train, seed=42)
rf_node = lp.train_random_forest(X_train_node, y_train, seed=42)

metrics_lr_node = lp.evaluate_model(lr_node, X_test_node, y_test, scaler_node)
metrics_rf_node = lp.evaluate_model(rf_node, X_test_node, y_test, None)

print(f"\n  LR  - AUC: {metrics_lr_node['auc_roc']:.4f} (Δ={metrics_lr_node['auc_roc']-metrics_lr_base['auc_roc']:+.4f})")
print(f"  RF  - AUC: {metrics_rf_node['auc_roc']:.4f} (Δ={metrics_rf_node['auc_roc']-metrics_rf_base['auc_roc']:+.4f})")

# =============================================================================
# 8. Experiment 4: +Temporal Features
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 4: +Temporal Features (32 features)")
print("="*80)

print("[8/11] Adding temporal features...")
temp_feat_train = add_temporal_features(G_1990_2015, train_pairs, cast_edges, 2015)
temp_feat_test = add_temporal_features(G_1990_2020, test_pairs, cast_edges, 2020)
X_train_temp = np.column_stack([X_train_base, temp_feat_train])
X_test_temp = np.column_stack([X_test_base, temp_feat_test])

lr_temp, scaler_temp = lp.train_logistic_regression(X_train_temp, y_train, seed=42)
rf_temp = lp.train_random_forest(X_train_temp, y_train, seed=42)

metrics_lr_temp = lp.evaluate_model(lr_temp, X_test_temp, y_test, scaler_temp)
metrics_rf_temp = lp.evaluate_model(rf_temp, X_test_temp, y_test, None)

print(f"\n  LR  - AUC: {metrics_lr_temp['auc_roc']:.4f} (Δ={metrics_lr_temp['auc_roc']-metrics_lr_base['auc_roc']:+.4f})")
print(f"  RF  - AUC: {metrics_rf_temp['auc_roc']:.4f} (Δ={metrics_rf_temp['auc_roc']-metrics_rf_base['auc_roc']:+.4f})")

# =============================================================================
# 9. Experiment 5: ALL COMBINED
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 5: ALL Combined (42 features)")
print("="*80)

print("[9/11] Combining all features...")
X_train_all = np.column_stack([X_train_base, ra_train, node_feat_train, temp_feat_train])
X_test_all = np.column_stack([X_test_base, ra_test, node_feat_test, temp_feat_test])

lr_all, scaler_all = lp.train_logistic_regression(X_train_all, y_train, seed=42)
rf_all = lp.train_random_forest(X_train_all, y_train, seed=42)

metrics_lr_all = lp.evaluate_model(lr_all, X_test_all, y_test, scaler_all)
metrics_rf_all = lp.evaluate_model(rf_all, X_test_all, y_test, None)

print(f"\n  LR  - AUC: {metrics_lr_all['auc_roc']:.4f} (Δ={metrics_lr_all['auc_roc']-metrics_lr_base['auc_roc']:+.4f})")
print(f"  RF  - AUC: {metrics_rf_all['auc_roc']:.4f} (Δ={metrics_rf_all['auc_roc']-metrics_rf_base['auc_roc']:+.4f})")

# =============================================================================
# 10. Experiment 6: XGBoost
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 6: XGBoost on ALL features")
print("="*80)

try:
    from xgboost import XGBClassifier
    
    print("[10/11] Training XGBoost...")
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42,
        n_jobs=-1, eval_metric='logloss'
    )
    xgb.fit(X_train_all, y_train)
    
    y_pred_proba_xgb = xgb.predict_proba(X_test_all)[:, 1]
    metrics_xgb = {'auc_roc': roc_auc_score(y_test, y_pred_proba_xgb)}
    
    print(f"\n  XGB - AUC: {metrics_xgb['auc_roc']:.4f} (Δ={metrics_xgb['auc_roc']-metrics_rf_base['auc_roc']:+.4f})")
    
except ImportError:
    print("  ⚠ XGBoost not installed. Skipping.")
    metrics_xgb = None

# =============================================================================
# 11. Summary
# =============================================================================
print("\n" + "="*80)
print(" " * 25 + "FINAL SUMMARY")
print("="*80)

results = []
results.append({'Experiment': '1. Baseline', 'Features': 24,
                'LR': metrics_lr_base['auc_roc'], 'RF': metrics_rf_base['auc_roc'],
                'Best': max(metrics_lr_base['auc_roc'], metrics_rf_base['auc_roc'])})
results.append({'Experiment': '2. +Resource Allocation', 'Features': 25,
                'LR': metrics_lr_ra['auc_roc'], 'RF': metrics_rf_ra['auc_roc'],
                'Best': max(metrics_lr_ra['auc_roc'], metrics_rf_ra['auc_roc'])})
results.append({'Experiment': '3. +Node Features', 'Features': 33,
                'LR': metrics_lr_node['auc_roc'], 'RF': metrics_rf_node['auc_roc'],
                'Best': max(metrics_lr_node['auc_roc'], metrics_rf_node['auc_roc'])})
results.append({'Experiment': '4. +Temporal Features', 'Features': 32,
                'LR': metrics_lr_temp['auc_roc'], 'RF': metrics_rf_temp['auc_roc'],
                'Best': max(metrics_lr_temp['auc_roc'], metrics_rf_temp['auc_roc'])})
results.append({'Experiment': '5. ALL Combined', 'Features': 42,
                'LR': metrics_lr_all['auc_roc'], 'RF': metrics_rf_all['auc_roc'],
                'Best': max(metrics_lr_all['auc_roc'], metrics_rf_all['auc_roc'])})
if metrics_xgb:
    results.append({'Experiment': '6. XGBoost', 'Features': 42,
                    'LR': np.nan, 'RF': np.nan, 'Best': metrics_xgb['auc_roc']})

df_results = pd.DataFrame(results)
df_results['Δ_Baseline'] = df_results['Best'] - df_results.loc[0, 'Best']

print("\n[11/11] Results:")
print(df_results.to_string(index=False))

best_idx = df_results['Best'].idxmax()
best = df_results.loc[best_idx]
print(f"\n🏆 WINNER: {best['Experiment']}")
print(f"   AUC-ROC: {best['Best']:.4f}")
print(f"   Improvement: +{best['Δ_Baseline']:.4f} ({best['Δ_Baseline']/df_results.loc[0,'Best']*100:.2f}%)")

# Save
df_results.to_csv('../data/processed/feature_experiments_results.csv', index=False)
print("\n💾 Saved: data/processed/feature_experiments_results.csv")
print("\n✅ ALL EXPERIMENTS COMPLETED!")
