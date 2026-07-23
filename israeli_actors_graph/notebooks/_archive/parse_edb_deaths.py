"""
Parse deceased actors from edb.co.il and match to our actor database
"""
import re
import pandas as pd
import pickle

# Load our graph to get actor list
with open('../data/processed/full_graph.pkl', 'rb') as f:
    G = pickle.load(f)

our_actors = set(G.nodes())
print(f'Our database: {len(our_actors)} actors')

# Deceased actors from edb.co.il (manually parsed)
# Format: Name | Years (birth-death)
edb_data = """
אריק איינשטיין|1939-2013
יעקב בן-סירא|1926-2016
רמי דנון|1942-2019
יהודה ברקן|1945-2020
יוסף שילוח|2011
חנה מרון|1923-2014
אסי דיין|1945-2014
אהרון איפלה|1951-2016
אורנה פורת|1923-2015
אריה אליאס|1921-2015
גדעון זינגר|1926-2015
עמוס לביא|1952-2010
ספי ריבלין|1947-2013
שמואל שילה|1929-2011
מוסקו אלקלעי|1931-2008
ישראל פוליאקוב|1941-2007
דודו טופז|1946-2009
יהודה פוקס|1918-2010
מישה אשרוב|1924-2003
דודיק סמדר|1930-2012
אברהם מור|1934-2012
רוזינה קמבוס|1950-2012
חיים חובא|1947-2012
אברהם דשא|1926-2004
יוסי יבלונקה|1946-2006
יוסי גרבר|1933-2016
יונה עטרי|1933-2019
בתיה לנצט|1921-2019
הלל נאמן|1928-2019
עודד תאומי|1937-2019
מרדכי ארנון|1941-2020
ניקו ניתאי|1930-2020
רוני פינקוביץ|1963-2020
גליה ישי|2020
יובל סטוניס|1983-2021
ליאור ייני|1936-2021
נולה צ'לטון|1922-2021
ראובן בר-יותם|1935-2021
שאול אליאס|1961-2021
טארק קופטי|1944-2022
זבולון מושאשוילי|1971-2023
תקוה מור|2025
רמה מסינגר|1968-2015
אמנון מסקין|1934-2015
מרגלית סטנדר|1938-2015
נחום שליט|1935-2015
עמוס שוב|1967-2015
יעקב בנאי|1918-1993
אברהם בן יוסף|1907-1980
אליעזר יונג|1928-2008
דן בן אמוץ|1923-1989
זאב ברלינסקי|1911-1989
אבנר חזקיהו|1926-1994
שמואל רודנסקי|1902-1989
נסים עזיקרי|1938-1989
טליה שפירא|1946-1992
דבורה בקון|1951-1983
רפי נלסון|1930-1988
שמעון בר|1926-1983
מוטי בהרב|1942-1989
מונה זילברשטיין|1947-1988
איציק גייר|1955-1988
משה איש כסית|1947-1987
רפאל קלצ'קין|1904-1987
זלמן לביוש|1907-1987
שרגא פרידמן|1923-1970
יאיר ואלין|1936-1963
יהושע לוף|1901-1985
יעקב טימן|1902-1974
מאיר מרגלית|1905-1973
אורנית זהבי|2019
אברהם הפנר|1934-2014
אברהם אבוטבול|1961-2012
אברהם רונאי|1932-2005
עדנה פלידל|1931-1993
עדי לב|1953-2005
עדה ואלרי טל|1935-1993
אוריאלה ווייט|1949-1993
אנלי הרפז|1972-2011
ג'וליאנו מר|1960-2011
גילי בן אוזיליו|1963-2009
גאולה נוני|1942-2014
ג'טה לוקה|1921-2001
דליה פן לרנר|1935-2011
דוידה קרול|1916-2011
דליה לביא|1940-2017
בוריס סבידנסקי|1940-2017
אריק לביא|1927-2004
אריאל פורמן|2016
רונית איבגי|2016
אביתר בורובסקי|2013
זהרירה חריפאי|2012
יובל זמיר|1963-2011
היינץ ברנרד|1923-1994
ברכה נאמן|1940-2004
ואדים קוטלרוב|1939-1998
מני פאר|1946-2014
מנשה ורשבסקי|1928-2001
מיכה לבינסון|1951-2017
רחל אטאס|1934-2004
רפי גינאי|1948-1995
שבתאי קונורטי|1942-2002
שלמה תרשיש|1947-2017
שמוליק סגל|1926-1997
שמוליק קראוס|1934-2013
נורית כהן|1959-2013
נחום בוכמן|1916-2007
נתן כוגן|1914-2009
עזריה רפופורט|1924-1997
עפרון אטקין|1952-2012
אלישבע מיכאלי|1928-2009
יוסי ידין|1920-2001
יהודה אפרוני|1930-2017
"""

# Parse data
deceased = {}
for line in edb_data.strip().split('\n'):
    if '|' not in line:
        continue
    parts = line.split('|')
    if len(parts) != 2:
        continue
    
    name = parts[0].strip()
    years = parts[1].strip()
    
    # Extract death year
    if '-' in years:
        death_year = int(years.split('-')[1])
    else:
        death_year = int(years)
    
    deceased[name] = death_year

print(f'\nParsed {len(deceased)} deceased actors from edb.co.il')

# Now match to our actors
# Convert names to Wikipedia format (replace spaces with _)
matched = {}
unmatched = []

for edb_name, death_year in deceased.items():
    # Try exact Wikipedia format
    wiki_name = edb_name.replace(' ', '_')
    
    if wiki_name in our_actors:
        matched[wiki_name] = death_year
    else:
        # Try variations
        found = False
        for actor in our_actors:
            actor_clean = actor.replace('_', ' ')
            if actor_clean == edb_name or edb_name in actor_clean or actor_clean in edb_name:
                matched[actor] = death_year
                found = True
                break
        
        if not found:
            unmatched.append(edb_name)

print(f'\n✅ Matched: {len(matched)} actors')
print(f'❌ Unmatched: {len(unmatched)} actors')

# Show distribution
years = list(matched.values())
if years:
    print(f'\n📅 Death year range: {min(years)}-{max(years)}')
    
    before_2015 = sum(1 for y in years if y <= 2015)
    period_2016_2020 = sum(1 for y in years if 2015 < y <= 2020)
    after_2020 = sum(1 for y in years if y > 2020)
    
    print(f'\n📊 By period:')
    print(f'   ≤2015 (train): {before_2015}')
    print(f'   2016-2020 (test): {period_2016_2020}')
    print(f'   >2020: {after_2020}')

# Save
df = pd.DataFrame([
    {'actor_slug': k, 'death_year': v}
    for k, v in matched.items()
])
df.to_csv('../data/processed/actor_death_years.csv', index=False)
print(f'\n💾 Saved to: data/processed/actor_death_years.csv')

# Show examples
print(f'\n🔍 Sample matched actors (recent):')
recent = [(k, v) for k, v in matched.items() if v >= 2015]
recent.sort(key=lambda x: x[1], reverse=True)
for actor, year in recent[:15]:
    print(f'   {actor.replace("_", " ")}: {year}')
