"""
Quick death date scraping - optimized version
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import pickle
import sys

def get_death_year(actor_slug):
    """Extract death year from Wikipedia categories"""
    if actor_slug.startswith('__name__:'):
        return None
    
    url = f'https://he.wikipedia.org/wiki/{actor_slug}'
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Educational/1.0'}, timeout=8)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check categories - most reliable
        categories = soup.find_all('div', id='mw-normal-catlinks')
        if categories:
            cat_text = ' '.join([c.get_text() for c in categories])
            
            # Look for "נפטרים ב-YYYY" or "שנפטרו ב-YYYY"
            match = re.search(r'(?:נפטרים|שנפטרו)\s+ב[-]?(\d{4})', cat_text)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2026:
                    return year
        
        # Check first paragraph for (YYYY-YYYY) pattern
        first_para = soup.find('p')
        if first_para:
            text = first_para.get_text()
            match = re.search(r'\((\d{4})\s*[-–]\s*(\d{4})\)', text)
            if match:
                birth = int(match.group(1))
                death = int(match.group(2))
                if birth < death <= 2026:
                    return death
        
        return None
    except:
        return None

print('🚀 Quick Death Date Scraping')
print('='*70)

# Load graph
with open('../data/processed/full_graph.pkl', 'rb') as f:
    G = pickle.load(f)

actors = [a for a in G.nodes() if not a.startswith('__name__:')]
print(f'Total actors to check: {len(actors)}')

# Test first
print('\n🧪 Testing on יעקב_בן-סירא (known deceased 2016)...')
test_year = get_death_year('יעקב_בן-סירא')
if test_year:
    print(f'✓ Test passed: found {test_year}')
else:
    print('✗ Test failed - check internet connection')
    sys.exit(1)

time.sleep(1)

# Full scraping with faster rate
print(f'\n⏳ Scraping all {len(actors)} actors (faster rate: 0.8s delay)...')
results = {}
errors = 0

for i, actor in enumerate(actors, 1):
    try:
        year = get_death_year(actor)
        if year:
            results[actor] = year
        
        # Progress every 25 actors
        if i % 25 == 0:
            print(f'   {i}/{len(actors)} ({100*i/len(actors):.1f}%) - Found: {len(results)} deceased')
        
        time.sleep(0.8)  # Faster than 1.5s
    except KeyboardInterrupt:
        print('\n⚠️  Interrupted by user')
        break
    except Exception as e:
        errors += 1

# Save results
print('\n' + '='*70)
print('📊 Results:')
print(f'   Actors checked: {i}')
print(f'   Deceased found: {len(results)}')
print(f'   Errors: {errors}')

if results:
    years = list(results.values())
    print(f'\n📅 Year range: {min(years)}-{max(years)}')
    
    # Distribution
    before_2015 = sum(1 for y in years if y <= 2015)
    period_2016_2020 = sum(1 for y in years if 2015 < y <= 2020)
    after_2020 = sum(1 for y in years if y > 2020)
    
    print(f'\n📊 By period:')
    print(f'   ≤2015: {before_2015}')
    print(f'   2016-2020: {period_2016_2020} ← Key for leakage test')
    print(f'   >2020: {after_2020}')
    
    # Recent examples
    recent = [(a, y) for a, y in results.items() if y >= 2015]
    recent.sort(key=lambda x: x[1], reverse=True)
    print(f'\n🔍 Recent deceased (2015+):')
    for actor, year in recent[:15]:
        print(f'   {actor.replace("_", " ")}: {year}')
    
    # Save
    df = pd.DataFrame([
        {'actor_slug': k, 'death_year': v}
        for k, v in results.items()
    ])
    df.to_csv('../data/processed/actor_death_years.csv', index=False)
    print(f'\n💾 Saved to: data/processed/actor_death_years.csv')
    print('✅ Done! Reload variable in notebook.')
else:
    print('⚠️  No deceased actors found')
