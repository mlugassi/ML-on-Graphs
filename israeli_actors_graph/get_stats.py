import pandas as pd

# Load data
cast = pd.read_csv('data/processed/cast_edges.csv')
movies = pd.read_csv('data/processed/movies.csv')
results = pd.read_csv('data/processed/link_prediction_results.csv')
period_metrics = pd.read_csv('data/processed/period_metrics.csv')

print("="*70)
print("BASIC DATA STATISTICS")
print("="*70)
print(f'Total movies: {len(movies)}')
print(f'Total cast edges: {len(cast)}')
print(f'Unique actors: {cast["actor_slug"].nunique()}')
print(f'Year range: {cast["year"].min()}-{cast["year"].max()}')
print(f'Movies with genres: {movies["genre"].notna().sum()} ({100*movies["genre"].notna().sum()/len(movies):.1f}%)')

print("\n" + "="*70)
print("PERIOD METRICS")
print("="*70)
print(period_metrics.to_string(index=False))

print("\n" + "="*70)
print("LINK PREDICTION RESULTS")
print("="*70)
print(results.to_string(index=False))

print("\n" + "="*70)
print("BEST MODEL")
print("="*70)
best = results.sort_values('AUC', ascending=False).iloc[0]
print(f"Configuration: {best['Experiment']}")
print(f"Model: {best['Model']}")
print(f"AUC: {best['AUC']:.4f}")
print(f"F1: {best['F1']:.4f}")
print(f"Precision: {best['Precision']:.4f}")
print(f"Recall: {best['Recall']:.4f}")
