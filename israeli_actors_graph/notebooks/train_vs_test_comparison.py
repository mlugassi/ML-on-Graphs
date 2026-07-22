"""
Feature Experiments - TRAIN vs TEST Comparison
================================================
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import roc_auc_score

sys.path.append('../src')
import link_prediction as lp
import link_features as lf

np.random.seed(42)

print("="*80)
print(" " * 15 + "TRAIN vs TEST COMPARISON - Feature Experiments")
print("="*80)

# Load data
cast_edges = pd.read_csv('../data/processed/cast_edges.csv')
G_1990_2015 = lp.build_graph_for_years(cast_edges, 1990, 2015)
G_1990_2020 = lp.build_graph_for_years(cast_edges, 1990, 2020)

# Get pairs
train_pos = lp.get_new_pairs(G_1990_2015, G_1990_2020)
train_neg = lp.sample_negatives(G_1990_2015, len(train_pos), exclude_pairs=set(train_pos), seed=42)
test_pos = lp.get_new_pairs(G_1990_2020, lp.build_graph_for_years(cast_edges, 1990, 2025))
test_neg = lp.sample_negatives(G_1990_2020, len(test_pos), exclude_pairs=set(test_pos), seed=42)

train_pairs = train_pos + train_neg
test_pairs = test_pos + test_neg
y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))
y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

print(f"\nTrain: {len(train_pos)} pos, {len(train_neg)} neg")
print(f"Test:  {len(test_pos)} pos, {len(test_neg)} neg")

# Helper functions (from run_experiments.py)
def add_resource_allocation(G, pairs):
    ra_map = {}
    for u, v, score in nx.resource_allocation_index(G, pairs):
        ra_map[(u, v)] = score
    return np.array([ra_map.get(p, 0.0) for p in pairs]).reshape(-1, 1)

def add_node_features(G, pairs):
    triangles = nx.triangles(G)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, max_iter=100)
    features = []
    for u, v in pairs:
        features.append([
            triangles.get(u, 0), triangles.get(v, 0),
            np.sqrt(triangles.get(u, 0) * triangles.get(v, 0)),
            clustering.get(u, 0.0), clustering.get(v, 0.0),
            (clustering.get(u, 0.0) + clustering.get(v, 0.0)) / 2,
            pagerank.get(u, 0.0), pagerank.get(v, 0.0),
            pagerank.get(u, 0.0) * pagerank.get(v, 0.0),
        ])
    return np.array(features)

def add_temporal_features(G, pairs, cast_edges, current_year):
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
        features.append([overlap, u_career, v_career, u_recent, v_recent, 
                        collab_count, len(u_years), len(v_years)])
    return np.array(features)

# =============================================================================
# Build features
# =============================================================================
print("\n" + "="*80)
print("Building features...")
print("="*80)

# Baseline
print("\n[1/4] Baseline features...")
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

# Temporal
print("[2/4] Temporal features...")
temp_feat_train = add_temporal_features(G_1990_2015, train_pairs, cast_edges, 2015)
temp_feat_test = add_temporal_features(G_1990_2020, test_pairs, cast_edges, 2020)
X_train_temp = np.column_stack([X_train_base, temp_feat_train])
X_test_temp = np.column_stack([X_test_base, temp_feat_test])

# ALL
print("[3/4] ALL features (Resource Allocation + Node + Temporal)...")
ra_train = add_resource_allocation(G_1990_2015, train_pairs)
ra_test = add_resource_allocation(G_1990_2020, test_pairs)
node_feat_train = add_node_features(G_1990_2015, train_pairs)
node_feat_test = add_node_features(G_1990_2020, test_pairs)
X_train_all = np.column_stack([X_train_base, ra_train, node_feat_train, temp_feat_train])
X_test_all = np.column_stack([X_test_base, ra_test, node_feat_test, temp_feat_test])

# =============================================================================
# Train and evaluate with BOTH train and test scores
# =============================================================================
print("\n[4/4] Training models...")
print("\n" + "="*80)
print(" " * 20 + "RESULTS: TRAIN vs TEST")
print("="*80)

results = []

# 1. Baseline
lr_base, scaler_base = lp.train_logistic_regression(X_train_base, y_train, seed=42)
rf_base = lp.train_random_forest(X_train_base, y_train, seed=42)

# Train scores
y_train_pred_lr_base = lr_base.predict_proba(scaler_base.transform(X_train_base))[:, 1]
y_train_pred_rf_base = rf_base.predict_proba(X_train_base)[:, 1]
train_auc_lr_base = roc_auc_score(y_train, y_train_pred_lr_base)
train_auc_rf_base = roc_auc_score(y_train, y_train_pred_rf_base)

# Test scores
y_test_pred_lr_base = lr_base.predict_proba(scaler_base.transform(X_test_base))[:, 1]
y_test_pred_rf_base = rf_base.predict_proba(X_test_base)[:, 1]
test_auc_lr_base = roc_auc_score(y_test, y_test_pred_lr_base)
test_auc_rf_base = roc_auc_score(y_test, y_test_pred_rf_base)

results.append({
    'Experiment': '1. Baseline (24 feat)',
    'LR_Train': train_auc_lr_base,
    'LR_Test': test_auc_lr_base,
    'LR_Gap': train_auc_lr_base - test_auc_lr_base,
    'RF_Train': train_auc_rf_base,
    'RF_Test': test_auc_rf_base,
    'RF_Gap': train_auc_rf_base - test_auc_rf_base
})

# 2. +Temporal
lr_temp, scaler_temp = lp.train_logistic_regression(X_train_temp, y_train, seed=42)
rf_temp = lp.train_random_forest(X_train_temp, y_train, seed=42)

y_train_pred_lr_temp = lr_temp.predict_proba(scaler_temp.transform(X_train_temp))[:, 1]
y_train_pred_rf_temp = rf_temp.predict_proba(X_train_temp)[:, 1]
train_auc_lr_temp = roc_auc_score(y_train, y_train_pred_lr_temp)
train_auc_rf_temp = roc_auc_score(y_train, y_train_pred_rf_temp)

y_test_pred_lr_temp = lr_temp.predict_proba(scaler_temp.transform(X_test_temp))[:, 1]
y_test_pred_rf_temp = rf_temp.predict_proba(X_test_temp)[:, 1]
test_auc_lr_temp = roc_auc_score(y_test, y_test_pred_lr_temp)
test_auc_rf_temp = roc_auc_score(y_test, y_test_pred_rf_temp)

results.append({
    'Experiment': '2. +Temporal (32 feat)',
    'LR_Train': train_auc_lr_temp,
    'LR_Test': test_auc_lr_temp,
    'LR_Gap': train_auc_lr_temp - test_auc_lr_temp,
    'RF_Train': train_auc_rf_temp,
    'RF_Test': test_auc_rf_temp,
    'RF_Gap': train_auc_rf_temp - test_auc_rf_temp
})

# 3. ALL Combined
lr_all, scaler_all = lp.train_logistic_regression(X_train_all, y_train, seed=42)
rf_all = lp.train_random_forest(X_train_all, y_train, seed=42)

y_train_pred_lr_all = lr_all.predict_proba(scaler_all.transform(X_train_all))[:, 1]
y_train_pred_rf_all = rf_all.predict_proba(X_train_all)[:, 1]
train_auc_lr_all = roc_auc_score(y_train, y_train_pred_lr_all)
train_auc_rf_all = roc_auc_score(y_train, y_train_pred_rf_all)

y_test_pred_lr_all = lr_all.predict_proba(scaler_all.transform(X_test_all))[:, 1]
y_test_pred_rf_all = rf_all.predict_proba(X_test_all)[:, 1]
test_auc_lr_all = roc_auc_score(y_test, y_test_pred_lr_all)
test_auc_rf_all = roc_auc_score(y_test, y_test_pred_rf_all)

results.append({
    'Experiment': '3. ALL Combined (42 feat)',
    'LR_Train': train_auc_lr_all,
    'LR_Test': test_auc_lr_all,
    'LR_Gap': train_auc_lr_all - test_auc_lr_all,
    'RF_Train': train_auc_rf_all,
    'RF_Test': test_auc_rf_all,
    'RF_Gap': train_auc_rf_all - test_auc_rf_all
})

# =============================================================================
# Display results
# =============================================================================
df_results = pd.DataFrame(results)

print("\n" + "─"*100)
print(f"{'Experiment':<30} {'LR Train':>10} {'LR Test':>10} {'LR Gap':>10} │ {'RF Train':>10} {'RF Test':>10} {'RF Gap':>10}")
print("─"*100)

for _, row in df_results.iterrows():
    print(f"{row['Experiment']:<30} {row['LR_Train']:>10.4f} {row['LR_Test']:>10.4f} "
          f"{row['LR_Gap']:>10.4f} │ {row['RF_Train']:>10.4f} {row['RF_Test']:>10.4f} "
          f"{row['RF_Gap']:>10.4f}")

print("─"*100)

# =============================================================================
# Analysis
# =============================================================================
print("\n" + "="*80)
print(" " * 30 + "ANALYSIS")
print("="*80)

print("\n📊 BASELINE (24 features):")
print(f"   LR:  Train={train_auc_lr_base:.4f}, Test={test_auc_lr_base:.4f}, Gap={train_auc_lr_base-test_auc_lr_base:.4f}")
print(f"   RF:  Train={train_auc_rf_base:.4f}, Test={test_auc_rf_base:.4f}, Gap={train_auc_rf_base-test_auc_rf_base:.4f}")
print(f"   → {'⚠️ Overfitting!' if train_auc_rf_base - test_auc_rf_base > 0.05 else '✅ Good generalization'}")

print("\n🚀 +TEMPORAL (32 features):")
print(f"   LR:  Train={train_auc_lr_temp:.4f}, Test={test_auc_lr_temp:.4f}, Gap={train_auc_lr_temp-test_auc_lr_temp:.4f}")
print(f"   RF:  Train={train_auc_rf_temp:.4f}, Test={test_auc_rf_temp:.4f}, Gap={train_auc_rf_temp-test_auc_rf_temp:.4f}")
print(f"   → {'⚠️ Overfitting!' if train_auc_rf_temp - test_auc_rf_temp > 0.05 else '✅ Good generalization'}")

print("\n🎯 ALL COMBINED (42 features):")
print(f"   LR:  Train={train_auc_lr_all:.4f}, Test={test_auc_lr_all:.4f}, Gap={train_auc_lr_all-test_auc_lr_all:.4f}")
print(f"   RF:  Train={train_auc_rf_all:.4f}, Test={test_auc_rf_all:.4f}, Gap={train_auc_rf_all-test_auc_rf_all:.4f}")
print(f"   → {'⚠️ Overfitting!' if train_auc_rf_all - test_auc_rf_all > 0.05 else '✅ Good generalization'}")

# Best TEST performer
best_test = max([
    ('Baseline LR', test_auc_lr_base),
    ('Baseline RF', test_auc_rf_base),
    ('Temporal LR', test_auc_lr_temp),
    ('Temporal RF', test_auc_rf_temp),
    ('ALL LR', test_auc_lr_all),
    ('ALL RF', test_auc_rf_all)
], key=lambda x: x[1])

print(f"\n🏆 BEST TEST PERFORMANCE: {best_test[0]} = {best_test[1]:.4f}")

# Save
df_results.to_csv('../data/processed/train_vs_test_comparison.csv', index=False)
print("\n💾 Saved: data/processed/train_vs_test_comparison.csv")
print("\n✅ Analysis complete!")
