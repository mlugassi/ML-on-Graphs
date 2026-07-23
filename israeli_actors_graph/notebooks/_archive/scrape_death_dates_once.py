"""
One-time scraping of actor death dates from Wikipedia.
Run this once, then use the saved CSV file.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import sys
sys.path.insert(0, '../src')

def get_death_year_improved(actor_slug):
    """
    Improved death year extraction from Hebrew Wikipedia.
    Returns death year (int) or None if alive/unknown.
    """
    if actor_slug.startswith('__name__:'):
        return None
    
    url = f'https://he.wikipedia.org/wiki/{actor_slug}'
    
    try:
        headers = {'User-Agent': 'IsraeliActorsResearch/1.0 (educational project)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Strategy 1: Look in infobox
        infobox = soup.find('table', class_='infobox')
        if infobox:
            # Get all rows
            for row in infobox.find_all('tr'):
                row_text = row.get_text()
                # Look for death-related keywords
                if any(keyword in row_text for keyword in ['נפטר', 'פטירה', 'מת', 'תאריך מוות']):
                    # Extract 4-digit years
                    years = re.findall(r'(19\d{2}|20\d{2})', row_text)
                    if years:
                        # Take the last year mentioned (usually the death year)
                        year = int(years[-1])
                        if 1900 <= year <= 2026:
                            return year
        
        # Strategy 2: Look for birth-death pattern in first paragraph
        first_para = soup.find('p')
        if first_para:
            text = first_para.get_text()
            # Pattern: (1950 - 2020) or (1950–2020)
            match = re.search(r'\((\d{4})\s*[-–]\s*(\d{4})\)', text)
            if match:
                birth_year = int(match.group(1))
                death_year = int(match.group(2))
                if birth_year < death_year and 1900 <= death_year <= 2026:
                    return death_year
        
        # Strategy 3: Check category for "נפטרים ב-YYYY"
        categories = soup.find_all('div', id='mw-normal-catlinks')
        if categories:
            cat_text = ' '.join([c.get_text() for c in categories])
            match = re.search(r'נפטרים ב[-]?(\d{4})', cat_text)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= 2026:
                    return year
        
        return None  # Assume alive
        
    except Exception as e:
        print(f'Error with {actor_slug}: {e}')
        return None


def main():
    print('🔍 ONE-TIME Death Date Collection')
    print('='*70)
    
    # Load existing graph to get actor list
    import pickle
    with open('../data/processed/full_graph.pkl', 'rb') as f:
        G = pickle.load(f)
    
    unique_actors = list(G.nodes())
    wiki_actors = [a for a in unique_actors if not a.startswith('__name__:')]
    
    print(f'Total unique actors: {len(unique_actors)}')
    print(f'Actors with Wikipedia links: {len(wiki_actors)}')
    
    # Known deceased actors for testing (manual verification)
    # אלה שחקנים ידועים שנפטרו - לבדיקה
    test_first = ['אריק_איינשטיין', 'חיים_טופול', 'שושנה_דמארי']
    
    print('\n🧪 Testing on known deceased actors first...')
    for actor in test_first:
        if actor in wiki_actors:
            year = get_death_year_improved(actor)
            actor_name = actor.replace('_', ' ')
            if year:
                print(f'   ✓ {actor_name}: {year}')
            else:
                print(f'   ✗ {actor_name}: NOT FOUND')
            time.sleep(1)
    
    print('\n' + '='*70)
    response = input('Continue with full scraping? (yes/no): ')
    if response.lower() != 'yes':
        print('Aborted.')
        return
    
    # Full scraping
    actor_death_years = {}
    errors = 0
    
    print('\n⏳ Scraping all actors (this will take ~15 minutes)...')
    for i, actor_slug in enumerate(wiki_actors, 1):
        if i % 50 == 0:
            print(f'   Progress: {i}/{len(wiki_actors)} ({100*i/len(wiki_actors):.1f}%) - Found {len(actor_death_years)} deceased')
        
        death_year = get_death_year_improved(actor_slug)
        if death_year is not None:
            actor_death_years[actor_slug] = death_year
        
        # Rate limiting
        time.sleep(1.0)
    
    # Results
    print('\n' + '='*70)
    print('📊 Final Results:')
    print(f'   Total actors checked: {len(wiki_actors)}')
    print(f'   Deceased actors found: {len(actor_death_years)}')
    print(f'   Assumed alive: {len(wiki_actors) - len(actor_death_years)}')
    
    if len(actor_death_years) > 0:
        years = list(actor_death_years.values())
        print(f'\n📅 Death year range: {min(years)}-{max(years)}')
        
        # Distribution by period
        died_before_2015 = sum(1 for y in years if y <= 2015)
        died_2016_2020 = sum(1 for y in years if 2015 < y <= 2020)
        died_after_2020 = sum(1 for y in years if y > 2020)
        
        print(f'\n📊 Distribution:')
        print(f'   ≤2015 (known in train): {died_before_2015}')
        print(f'   2016-2020 (test period): {died_2016_2020}')
        print(f'   >2020 (recent): {died_after_2020}')
        
        # Show examples
        print('\n🔍 Sample deceased actors:')
        samples = sorted(actor_death_years.items(), key=lambda x: x[1], reverse=True)[:15]
        for actor, year in samples:
            actor_name = actor.replace('_', ' ')
            print(f'   • {actor_name}: {year}')
    
    # Save to CSV
    death_df = pd.DataFrame([
        {'actor_slug': k, 'death_year': v} 
        for k, v in actor_death_years.items()
    ])
    death_df.to_csv('../data/processed/actor_death_years.csv', index=False)
    print(f'\n💾 Saved to: data/processed/actor_death_years.csv')
    print('✅ Complete! Now the notebook can load this file.')


if __name__ == '__main__':
    main()
