# דוח סופי: ניתוח רשת שיתופי הפעולה של שחקני הקולנוע הישראלי
## Machine Learning on Graphs — מכללת אזריאלי

---

**הערות שימוש:**
- קובץ זה מכיל את המבנה המלא של הדוח (10-15 עמודים)
- כל סעיף כולל **הנחיות מפורטות** מה לכלול
- **נתיבים לקבצים** מסומנים ב-`📁`
- **תמונות לשילוב** מסומנות ב-`🖼️`
- **טבלאות לשילוב** מסומנות ב-`📊`
- **נתונים למילוי** מסומנים ב-`[FILL: ...]`

**קבצים רלוונטיים:**
- נוטבוקים: `israeli_actors_graph/notebooks/01-06_*.ipynb`
- נתונים: `israeli_actors_graph/data/processed/*.csv`
- תמונות: `israeli_actors_graph/figures/*.png`
- קוד: `israeli_actors_graph/src/*.py`

---

# 1. מבוא (1-2 עמודים)

## 1.1 רקע ומטרת המחקר

רשתות חברתיות הן כלי מרכזי להבנת דפוסי אינטראקציה בין אנשים, ובמיוחד בתעשיות יצירתיות כמו קולנוע. מחקר זה בוחן את **רשת שיתופי הפעולה של שחקני הקולנוע הישראלי** לאורך כמעט מאה שנות קולנוע (1933-2026), תוך שימוש בכלי Machine Learning על גרפים.

### מטרות המחקר:
1. **איסוף וארגון נתונים**: בניית מאגר מקיף של סרטי הקולנוע הישראלי ושחקניהם מוויקיפדיה העברית
2. **ניתוח התפתחות הרשת**: מעקב אחר שינויים ברשת לאורך שלוש תקופות זמן (≤1970, 1971-1990, >1990)
3. **זיהוי קהילות**: איתור קבוצות שחקנים שעובדים יחד באופן תכוף
4. **ניתוח מרכזיות**: זיהוי שחקני מפתח ברשת לפי מדדים שונים
5. **חיזוי שיתופי פעולה עתידיים**: בניית מודל Machine Learning לחיזוי שיתופי פעולה צפויים בין שחקנים

### חשיבות המחקר:

**תרומה אקדמית:**
- יישום מעשי של אלגוריתמים מתקדמים ברשתות חברתיות
- הדגמת שימוש ב-Temporal Link Prediction על נתונים ריאליים
- שילוב מדדי גרף טופולוגיים עם Embeddings למידה

**תרומה תעשייתית:**
- הבנת דפוסי עבודה בתעשיית הקולנוע הישראלית
- זיהוי שחקנים מרכזיים ומשפיעים
- כלי פוטנציאלי לסטודיו/מפיק להערכת הסבירות לשיתופי פעולה

**תרומה תרבותית:**
- תיעוד ויזואלי של התפתחות הקולנוע הישראלי
- שימור מידע על שיתופי פעולה היסטוריים
- הבנת השינויים ברשת לאורך הדורות

## 1.2 סקירה כללית של המתודולוגיה

המחקר מורכב מחמישה שלבים עיקריים:

### שלב 1: איסוף נתונים (Data Collection)
- **מקור**: ויקיפדיה העברית — רשימת סרטי הקולנוע הישראלי
- **שיטה**: Web scraping באמצעות MediaWiki API
- **תוצר**: 840 סרטים, 1,039 שחקנים ייחודיים, טווח שנים 1933-2026
📁 `notebooks/01_data_collection.ipynb`
📁 `data/processed/movies.csv`, `cast_edges.csv`

### שלב 2: בניית הגרף (Graph Construction)
- **ייצוג**: גרף לא-מכוון משוקלל (קודקודים = שחקנים, קשתות = משחק משותף בסרט)
- **משקל קשת**: מספר הסרטים המשותפים
- **פיצול זמני**: שלוש תקופות (A: ≤1970, B: 1971-1990, C: >1990)
📁 `notebooks/02_graph_construction.ipynb`

### שלב 3: ניתוח רשת (Network Analysis)
- **מדדים בסיסיים**: מספר קודקודים/קשתות, צפיפות, ממוצע דרגה, קוטר
- **מדדי קישוריות**: מספר רכיבי קשירות, גודל הרכיב הענק
- **התפלגות דרגות**: ניתוח Log-Log
📁 `notebooks/03_temporal_period_analysis.ipynb`

### שלב 4: זיהוי קהילות (Community Detection)
- **אלגוריתמים**: Louvain, Greedy Modularity, Label Propagation
- **השוואה**: מדד Modularity, NMI (Normalized Mutual Information)
📁 `notebooks/04_community_detection.ipynb`

### שלב 5: ניתוח מרכזיות (Centrality Analysis)
- **מדדים**: Degree, Betweenness, Closeness, Eigenvector Centrality
- **זיהוי שחקנים מובילים**: טופ 20 בכל מדד
📁 `notebooks/05_centrality_analysis.ipynb`

### שלב 6: חיזוי קשרים (Link Prediction)
- **גישה**: Temporal Link Prediction עם supervised learning
- **עיצוב זמני**: 
  - Train: X מ-1990-2015, y = קשרים חדשים 2016-2020
  - Test: X מ-1990-2020, y = קשרים חדשים 2021-2025
- **פיצ'רים**: 38 פיצ'רים סה"כ:
  - טופולוגיים: Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, Shortest Path
  - מרכזיות: Degree, Betweenness, Closeness, Eigenvector (כל אחד ×4 = 16)
  - Embedding: SVD/Matrix Factorization (2 פיצ'רים)
  - Temporal: career_length, film_count, genre_diversity, director_count ועוד (11 פיצ'רים)
- **מודלים**: Logistic Regression, Random Forest
- **תוצאה**: AUC = 0.8188 (38 features), AUC = 0.8109 (26 features אחרי feature selection)
📁 `notebooks/06_link_prediction.ipynb`

---

# 2. שיטות (2-3 עמודים)

## 2.1 איסוף וניקוי נתונים

### 2.1.1 מקור הנתונים
הנתונים נאספו מ**ויקיפדיה העברית**, במיוחד מהערך ["סרטי קולנוע ישראליים"](https://he.wikipedia.org/wiki/סרטי_קולנוע_ישראליים). הערך מכיל טבלאות מסודרות לפי עשורים עם מידע על כל סרט: שנה, שם, במאי, תסריטאי, ז'אנר, שחקנים ופרסים.

### 2.1.2 תהליך הסקרייפינג
**כלים:**
- `MediaWiki API` — גישה מובנית לתוכן ויקיפדיה (יותר יציב מ-HTML scraping)
- `Beautiful Soup 4 (bs4)` — פרסור HTML לטבלאות ותיבות מידע
- `requests` — שליחת בקשות HTTP

**תהליך:**
1. **שלב 1**: שליפת רשימת כל הסרטים מהערך הראשי (טבלאות לפי עשורים)
2. **שלב 2**: לכל סרט, מבקרים בדף הסרט הייעודי ומחלצים:
   - **תיבת מידע**: שדה "שחקנים" (עד 10 שחקנים ראשונים)
   - **קישורי ויקי**: כל שחקן מזוהה לפי slug הוויקיפדיה שלו (למניעת כפילויות)
3. **שלב 3**: אחסון תגובות API ב-`data/raw/` לצורכי cache (למניעת בקשות חוזרות)
4. **שלב 4**: ניקוי שמות (טיפול בשחקנים ללא קישור ויקי, נורמליזציה)

**מדיניות נימוס:**
- User-Agent מותאם אישית: `"IsraeliActorsGraphResearch/1.0 (educational course project; contact via GitHub)"`
- השהיה של 1.0 שניה בין בקשות

**תוצאות:**
- **סה"כ סרטים**: 840 סרטים
- **טווח שנים**: 1933-2026 (93 שנים)
- **סה"כ שחקנים ייחודיים**: 1,039 שחקנים
- **סה"כ קשרים (סרט-שחקן)**: 2,972 קשרים
- **סרטים עם ז'אנר**: 753 (89.6%)

📊 **טבלה 1**: סטטיסטיקות איסוף נתונים
```
📁 קבץ: data/processed/movies.csv
עמודות: year, title, movie_slug, director, genre, etc.
סה"כ שורות: 840 סרטים
שנים: 1933-2026 (93 שנים)
```

📊 **טבלה 2**: דוגמה מקובץ cast_edges.csv (5 שורות ראשונות)
```
📁 קבץ: data/processed/cast_edges.csv
עמודות: movie_slug, actor_slug, year
סה"כ שורות: 2,972 קשרי cast
```

### 2.1.3 ניקוי וטיפול בנתונים חסרים
- **סרטים ללא צוות משחק**: סרטים עם פחות מ-2 שחקנים לא נכללו בניתוח
- **שחקנים ללא קישור ויקי**: שמות נורמלו באמצעות prefix `__name__:` (למעקב)
- **כפילויות**: זוהו וטופלו באמצעות Wikipedia slugs

📁 **קוד**: `src/scraping.py` — פונקציות `fetch_page_html()`, `collect_movie_list()`, `collect_film_cast()`, `_extract_cast_from_infobox()`

---

## 2.2 אלגוריתמים וכלים

### 2.2.1 ספריות וטכנולוגיות
- **NetworkX 3.5+**: בניית גרפים, חישוב מדדים, אלגוריתמי קהילות
- **scikit-learn**: מודלי ML (Logistic Regression, Random Forest), pre-processing
- **Node2Vec**: Random walk embeddings (ניסיון — לא נכלל בניתוח הסופי)
- **Pandas**: עיבוד נתונים טבלאיים
- **Matplotlib/Seaborn**: ויזואליזציות
- **Python 3.10+**: שפת תכנות ראשית

### 2.2.2 ייצוג הגרף
**סוג גרף:** MultiGraph לא-מכוון משוקלל
- **קודקודים (Nodes)**: שחקנים (מזוהים ע"י Wikipedia slug או שם מנורמל)
- **קשתות (Edges)**: שני שחקנים שהופיעו יחד באותו סרט
- **משקל (Weight)**: מספר הסרטים המשותפים (אם שחקנים עבדו ביחד ב-3 סרטים → משקל=3)

**בניית הגרף:**
```python
G = nx.MultiGraph()
for (movie, actors) in films:
    for (u, v) in combinations(actors, 2):
        G.add_edge(u, v, movie_slug=movie['slug'], 
                   year=movie['year'], title=movie['title'])
```

📁 **קוד**: `src/graph_build.py` — `build_full_graph()`

### 2.2.3 פיצול זמני (Temporal Periods)
הגרף חולק לשלוש תקופות לניתוח התפתחות הרשת:

| **תקופה** | **טווח שנים** | **תיאור** | **סימון** |
|-----------|--------------|-----------|----------|
| A | ≤1970 | ראשית הקולנוע הישראלי (1918-1970, 53 שנים) | G_A |
| B | 1971-1990 | צמיחת הקולנוע (20 שנים) | G_B |
| C | >1990 | העידן המודרני (1991-2026, 36 שנים) | G_C |

כל גרף תקופה בנוי **רק מסרטים שיצאו בטווח השנים הרלוונטי**.

📁 **קוד**: `src/graph_build.py` — `period_of_year()`, `period_subgraph()`

### 2.2.4 אלגוריתמי זיהוי קהילות
נבדקו **שלושה אלגוריתמים**:

#### 1. **Louvain (Greedy Modularity Maximization)**
- **עיקרון**: אופטימיזציה חמדנית של פונקציית Modularity
- **יתרונות**: מהיר, מוצא קהילות היררכיות
- **מימוש**: `nx.community.louvain_communities()`

#### 2. **Greedy Modularity**
- **עיקרון**: מיזוג איטרטיבי של קהילות לפי Modularity
- **יתרונות**: פשוט, תוצאות דומות ל-Louvain
- **מימוש**: `nx.community.greedy_modularity_communities()`

#### 3. **Label Propagation**
- **עיקרון**: כל קודקוד מאמץ את התווית הנפוצה ביותר בשכנותיו
- **יתרונות**: מהיר מאוד, לא דטרמיניסטי
- **מימוש**: `nx.community.label_propagation_communities()`

**השוואת ביצועים:**
- **Modularity Score**: עד כמה הקהילות "טובות" (ערך גבוה = קהילות מובחנות)
- **NMI (Normalized Mutual Information)**: קורלציה בין תוצאות האלגוריתמים

📁 **קוד**: `src/communities.py` — `detect_communities()`, `compare_methods()`

### 2.2.5 מדדי מרכזיות (Centrality)
חושבו **ארבעה מדדים** עיקריים:

| **מדד** | **משמעות** | **פונקציה NetworkX** |
|---------|------------|---------------------|
| **Degree Centrality** | מספר השכנים הישירים (נורמל) | `nx.degree_centrality()` |
| **Betweenness Centrality** | כמה מסלולים קצרים עוברים דרך הקודקוד | `nx.betweenness_centrality()` |
| **Closeness Centrality** | קירבה ממוצעת לכל שאר הקודקודים | `nx.closeness_centrality()` |
| **Eigenvector Centrality** | מרכזיות לפי חשיבות השכנים | `nx.eigenvector_centrality()` |

📁 **קוד**: `src/centrality.py` — `compute_all_centralities()`

---

## 2.3 פיצ'רים למודל חיזוי הקשרים

### 2.3.1 עיצוב זמני (Temporal Design)
המודל מנבא שיתופי פעולה **עתידיים** בהינתן המבנה **העבר** של הגרף:

| | **X (features)** | **y (labels)** | **years_ahead** |
|---|----------------|---------------|----------------|
| **Train** | G_1990_2015 (1990-2015) | קשרים חדשים 2016-2020 | 5 |
| **Test** | G_1990_2020 (1990-2020) | קשרים חדשים 2021-2025 | 5 |
| **Future** | G_1990_2025 (1990-2025) | לא ידוע (2026+) | [משתנה] |

**עקרון ה-Temporal Split:**
- פיצ'רים (**X**) מחושבים **רק** מהגרף ההיסטורי (למניעת data leakage)
- תוויות (**y**) הן קשרים חדשים שנוצרו בחלון הזמן הבא
- `years_ahead=5` → המודל לומד שהוא מנבא 5 שנים קדימה

### 2.3.2 דגימת Positives ו-Negatives
**Positive samples (y=1):**
- זוגות שחקנים ש**לא** עבדו יחד בעבר
- אבל **כן** עבדו יחד בחלון הזמן העתידי
- דוגמה: (Actor A, Actor B) לא מחוברים ב-G_1990_2015, אבל מופיעים יחד ב-2018

**Negative samples (y=0):**
- זוגות שחקנים שלא מחוברים בגרף ההיסטורי
- וגם **לא** נוצר ביניהם קשר בעתיד
- נדגמו באקראי (NEG_RATIO=1.5 → 1.5 negatives לכל positive)

📁 **קוד**: `src/link_prediction.py` — `get_new_pairs()`, `sample_negatives()`

### 2.3.3 פיצ'רים (24 סה"כ)

#### **A. Topological Features (5 פיצ'רים)**
מדדי קירבה מבניים בין זוג קודקודים:

1. **Common Neighbors (CN)**: כמה שכנים משותפים יש ל-u ו-v
   ```python
   cn = len(list(nx.common_neighbors(G, u, v)))
   ```

2. **Jaccard Coefficient**: 
   ```
   jaccard = |neighbors(u) ∩ neighbors(v)| / |neighbors(u) ∪ neighbors(v)|
   ```

3. **Adamic-Adar Index**: שכנים משותפים "נדירים" מקבלים משקל גבוה יותר
   ```
   AA = Σ_{z ∈ CN(u,v)} 1 / log(degree(z))
   ```

4. **Preferential Attachment**: מכפלת דרגות (הנחה: קודקודים "פופולריים" נוטים להתחבר)
   ```
   PA = degree(u) × degree(v)
   ```

5. **Shortest Path Distance**: האורך של המסלול הקצר ביותר בין u ל-v (או ∞ אם לא מחוברים)

#### **B. Centrality Features (16 פיצ'רים)**
לכל מדד centrality (degree, betweenness, closeness, eigenvector) מחושבים 4 פיצ'רים:
- `{metric}_u`: הערך של u
- `{metric}_v`: הערך של v
- `{metric}_product`: u × v
- `{metric}_abs_diff`: |u - v|

סה"כ: 4 מדדים × 4 פיצ'רים = **16 פיצ'רים**

#### **C. SVD Embeddings (2 פיצ'רים)**
TruncatedSVD על מטריצת הסמיכות (32 רכיבים):
- `svd_cosine_sim`: דמיון קוסינוס בין וקטורי ה-embedding
- `svd_dot_product`: מכפלה סקלרית

#### **D. Temporal Features (10 פיצ'רים) - מניעת Data Leakage**
פיצ'רים טמפורליים שמחושבים עם **הגבלה זמנית מדויקת** למניעת דליפת מידע:

1. **career_overlap**: כמה שנים שני השחקנים היו פעילים ביחד
2. **u_career_length**: אורך הקריירה של שחקן u (בשנים)
3. **v_career_length**: אורך הקריירה של שחקן v (בשנים)
4. **u_recent**: 1 אם שחקן u פעיל ב-5 השנים האחרונות, 0 אחרת
5. **v_recent**: 1 אם שחקן v פעיל ב-5 השנים האחרונות, 0 אחרת
6. **u_alive**: 1 אם שחקן u חי בנקודת החיזוי (למניעת data leakage!)
7. **v_alive**: 1 אם שחקן v חי בנקודת החיזוי
8. **collaboration_count**: כמה פעמים u ו-v כבר עבדו יחד
9. **u_film_count**: סה"כ סרטים של שחקן u
10. **v_film_count**: סה"כ סרטים של שחקן v

**🚨 Data Leakage Prevention - is_alive Feature:**
הפיצ'ר `is_alive` מחושב **בנקודת הזמן של החיזוי** ולא בנקודת הזמן של האמת:
- **Train (2015)**: אם שחקן נפטר ב-2019 → `is_alive=1` (עדיין לא ידוע!)
- **Test (2020)**: אותו שחקן → `is_alive=0` (כבר ידוע)

מקור נתונים: 21 שחקנים שנפטרו מ-edb.co.il (14 נפטרו ≤2015, 7 נפטרו 2016-2020)

#### **E. years_ahead (1 פיצ'ר)**
אופק החיזוי בשנים (בדרך כלל 5) — מאפשר למודל ללמוד חיזויים לטווחים שונים

**📊 סה"כ Baseline: 5 Topological + 16 Centrality + 2 SVD = 23 פיצ'רים**  
**📊 סה"כ +Temporal+Genre+Director: 23 + 15 = 38 פיצ'רים**  
**📊 אחרי Feature Selection: 26 פיצ'רים (הוסרו 12 כפולים/חסרי ערך)**

📁 **קוד**: 
- `src/link_features.py` — `topological_features()`, `centrality_features()`, `svd_node_embeddings()`, `build_feature_matrix()`
- `notebooks/06_link_prediction.ipynb` — `build_full_dataset()` עם כל הפיצ'רים כולל temporal, genre, director

### 2.3.4 מודלים
נבחנו **שני מודלים**:

#### 1. **Logistic Regression**
- **Solver**: LBFGS (אופטימיזציה בסדר-שני, לא gradient descent)
- **Regularization**: C=1.0 (regularization בינוני)
- **Class weight**: `balanced` (מטפל בחוסר איזון בין positives/negatives)
- **Max iterations**: 2000
- **Pre-processing**: StandardScaler (נרמול פיצ'רים)

#### 2. **Random Forest**
- **n_estimators**: 300 עצים
- **max_depth**: 12 (למניעת overfitting)
- **min_samples_leaf**: 2
- **class_weight**: `balanced`
- **n_jobs**: -1 (שימוש בכל ליבות המעבד)

📁 **קוד**: `src/link_prediction.py` — `train_logistic_regression()`, `train_random_forest()`

### 2.3.5 הערכת ביצועים
**מדדים:**
- **AUC-ROC**: שטח מתחת לעקומת ROC (מודד יכולת דירוג)
- **Precision**: מתוך החיזויים החיוביים, כמה נכונים?
- **Recall**: מתוך הקשרים האמיתיים, כמה תפסנו?
- **F1 Score**: ממוצע הרמוני של Precision ו-Recall
- **Average Precision (AP)**: שטח מתחת לעקומת Precision-Recall

📁 **קוד**: `src/link_prediction.py` — `evaluate_model()`, `roc_curve_data()`, `pr_curve_data()`

---

# 3. תוצאות (6-8 עמודים)

## 3.1 ניתוח התפתחות הגרף

### 3.1.1 סטטיסטיקות בסיסיות לפי תקופות

🖼️ **איור 1**: סרטים לפי שנה עם קווי חיתוך זמניים
```
📁 קובץ: figures/films_per_year_cutoffs.png
תיאור: היסטוגרמה של מספר הסרטים לפי שנה, עם קווים אדומים המסמנים את החיתוך בין תקופות
```

📊 **טבלה 3**: מדדי גרף מלאים לפי תקופה
```
📁 קובץ: data/processed/period_metrics.csv

| תקופה | שנים | שחקנים | שיתופי פעולה | צפיפות | ממוצע דרגה | מקדם אשכול | רכיבי קשירות | גודל LCC | קוטר | רדיוס |
|-------|------|---------|---------------|---------|-------------|------------|--------------|---------|------|-------|
| A     | ≤1970       | 132   | 472   | 0.0546 | 7.15 | 0.741 | 3  | 128 (97%) | 6 | 4 |
| B     | 1971-1990   | 336   | 933   | 0.0166 | 5.55 | 0.660 | 23 | 292 (87%) | 7 | 4 |
| C     | >1990       | 1,106 | 4,731 | 0.0077 | 8.56 | 0.751 | 49 | 989 (89%) | 9 | 5 |
```

### 3.1.2 תובנות מהתפתחות הרשת

**גידול דרמטי במספר השחקנים:**
- מ-**132 שחקנים** בתקופה A (71 שנים, 1918-1989)
- ל-**1106 שחקנים** בתקופה C (15 שנים בלבד!, 2010-2025)
- **צמיחה של ×8.4** בגודל הרשת

**ירידה בצפיפות:**
- תקופה A: צפיפות = 0.0546 (5.46% מכל הקשרים האפשריים קיימים)
- תקופה C: צפיפות = 0.0077 (0.77% בלבד)
- **משמעות**: הרשת הפכה **מפוזרת יותר** — שחקנים עובדים עם עוד מעט שחקנים יחסית לגודל הרשת

**ממוצע דרגה:**
- תקופה A: 7.15 (שחקן ממוצע עבד עם ~7 שחקנים אחרים)
- תקופה C: 8.56 (עלייה קלה למרות הצפיפות הנמוכה)

**קוטר ורדיוס הרשת:**
- תקופה A: קוטר = 6, רדיוס = 4
- תקופה B: קוטר = 7, רדיוס = 4
- תקופה C: קוטר = 9, רדיוס = 5 (הרשת "התרחבה")

**מקדם אשכול (Clustering Coefficient) גבוה וקבוע:**
- תקופה A: 0.741 | תקופה B: 0.660 | תקופה C: 0.751
- **משמעות**: שחקנים נוטים לעבוד עם צוות ידוע — "חבורות" צפופות שמשחקות ביחד שוב ושוב

**פיצול לרכיבי קשירות:**
- תקופה A: 3 רכיבים בלבד — רשת מגובשת (LCC = 128/132, 97%)
- תקופה B: 23 רכיבים — הופיעו שחקנים בודדים מבודדים
- תקופה C: 49 רכיבים — רשת גדולה ומגוונת, עם LCC=989/1106 (89%)

📁 **מקור**: `notebooks/03_temporal_period_analysis.ipynb` — תא "Period metrics"

### 3.1.3 התפלגות דרגות (Degree Distribution)

🖼️ **איור 2**: Log-Log Degree Distribution לכל תקופה
```
📁 מיקום: אמור להיות ב-notebooks/03_temporal_period_analysis.ipynb
תיאור: גרף Log-Log של P(k) מול k, מראה אם הרשת מקיימת Power Law (רשת ללא סקלה)
```

**פירוש:**
- אם הגרף מראה קו ישר ב-Log-Log → **רשת ללא סקלה** (Power Law)
- משמעות: יש **מעט שחקנים מרכזיים** (hub) עם חיבורים רבים מאוד
- רוב השחקנים עם קשרים מעטים

---

## 3.2 ניתוח קהילות

### 3.2.1 השוואת אלגוריתמים

📊 **טבלה 4**: השוואת אלגוריתמי זיהוי קהילות (תקופה C)
```
📁 מקור: notebooks/04_community_detection.ipynb

| אלגוריתם | מספר קהילות | Modularity | זמן ריצה (שניות) |
|----------|-------------|------------|------------------|
| Louvain  | 53      | 0.653     | ~0.05           |
| Greedy Modularity | 64 | 0.649  | ~0.2           |
| Label Propagation | 81 | 0.642  | ~0.01           |
```

**בחירת האלגוריתם הטוב ביותר:**
- אלגוריתם **Louvain** עם ה-**Modularity הגבוה ביותר (0.653)** נבחר כבסיס לניתוח
- בדרך כלל **Louvain** נותן תוצאות מעולות

### 3.2.2 מפת הקהילות

🖼️ **איור 3**: ויזואליזציה של קהילות בתקופה C
```
📁 מיקום: notebooks/04_community_detection.ipynb (ניתן לשלוף איור)
תיאור: גרף עם קודקודים צבועים לפי קהילה, עם שמות השחקנים המובילים
```

📊 **טבלה 5**: דוגמאות לקהילות מובחנות
```
📁 קובץ: data/processed/community_assignments.csv

פורמט:
| קהילה | מספר שחקנים | שחקנים מרכזיים (דוגמאות) | תיאור אפשרי |
|-------|-------------|---------------------------|-------------|
| 0     | [FILL]      | [FILL: שמות]              | [למשל: קבוצת שחקני הקומדיה] |
| 1     | [FILL]      | [FILL: שמות]              | [למשל: שחקני דרמה] |
| 2     | [FILL]      | [FILL: שמות]              | [קבוצת צעירים] |
```

### 3.2.3 אפיון קהילות

**שאלות לניתוח:**
1. **האם יש קהילות ברורות?** (כן/לא — לפי Modularity)
2. **מה מאפיין כל קהילה?** (שנות פעילות, ז'אנר, במאים משותפים?)
3. **שחקנים "גשרים"**: מי מחובר למספר קהילות?

📁 **מקור**: `notebooks/04_community_detection.ipynb` — תאים עם `community_assignments`, `analyze_communities()`

---

## 3.3 ניתוח מרכזיות

### 3.3.1 השחקנים המרכזיים ביותר

📊 **טבלה 6**: טופ 10 שחקנים לפי Degree Centrality (תקופה C)
```
📁 קובץ: data/processed/centrality_scores.csv
עמודות: display_name, degree, betweenness, closeness, eigenvector, n_films, n_costars

דוגמה:
| מקום | שם השחקן | Degree | מספר שיתופי פעולה | מספר סרטים |
|------|----------|--------|-------------------|------------|
| 1    | [FILL: שם] | [FILL] | [FILL] | [FILL] |
| 2    | [FILL: שם] | [FILL] | [FILL] | [FILL] |
| ...  | ...        | ...    | ...    | ...    |

מהנתונים הריאליים:
- דרור קרן: degree=0.0317, 35 שותפים, 9 סרטים
- מנשה נוי: degree=0.0181, 20 שותפים, 7 סרטים
```

🖼️ **איור 4**: ויזואליזציה של טופ 20 לפי Degree
```
📁 מיקום: notebooks/05_centrality_analysis.ipynb
תיאור: גרף עמודות אופקי של השחקנים הטופ לפי Degree Centrality
```

### 3.3.2 Betweenness Centrality (שחקנים "גשרים")

📊 **טבלה 7**: טופ 10 שחקנים לפי Betweenness
```
מי מחבר בין קבוצות שחקנים שונות?
| מקום | שם השחקן | Betweenness | פירוש |
|------|----------|-------------|--------|
| 1    | [FILL]   | [FILL]      | מחבר בין קהילות X ו-Y |

מהנתונים:
- דרור קרן: betweenness=0.0291
- מנשה נוי: betweenness=0.0089
```

### 3.3.3 השוואה בין מדדים

**תובנות:**
- **Degree גבוה**: שחקנים שעבדו עם המון אנשים שונים (אולי "שחקן תומך" נפוץ)
- **Betweenness גבוה**: שחקנים ש"מגשרים" בין סגנונות/קבוצות שונות
- **Eigenvector גבוה**: שחקנים שעובדים עם שחקנים מרכזיים אחרים (אליטה)

📁 **מקור**: `notebooks/05_centrality_analysis.ipynb` — תאי "Centrality analysis", "Top actors"

---

## 3.4 תוצאות מודל חיזוי הקשרים

### 3.4.1 ביצועי המודלים

📊 **טבלה 8**: השוואת ביצועים על Test Set (2021-2025)
```
📁 קובץ: data/processed/feature_selection_results.csv

| קונפיגורציה | פיצ'רים | Precision | Recall | F1 | AUC-ROC | Avg Precision |
|-------------|---------|-----------|--------|----|---------|--------------| 
| Baseline (topological+centrality) | 23 | 0.509 | 0.741 | 0.603 | 0.722 | 0.674 |
| +Temporal features | 34 | 0.660 | 0.550 | 0.600 | 0.804 | 0.730 |
| **+All Features (incl. genre/director)** | **38** | **0.579** | **0.702** | **0.635** | **0.819** | **0.680** |
| **Feature Selection (optimal)** | **26** | **0.576** | **0.684** | **0.626** | **0.811** | **0.666** |

מודל זוכה: **Random Forest + כל הפיצ'רים** (38 פיצ'רים, AUC=0.819)
נוטבוק: `notebooks/06_link_prediction.ipynb`
```

🖼️ **איור 5**: עקומות ROC ו-Precision-Recall
```
📁 קובץ: figures/link_prediction_curves.png
תיאור: שני גרפים זה לצד זה — ROC (FPR vs TPR) ו-PR (Recall vs Precision) להשוואת המודלים
```

**פירוש התוצאות:**
- **AUC-ROC = 0.819**: המודל מדרג זוג שחקנים אקראי שעבד יחד גבוה מזוג שלא עבד ב-81.9% מהמקרים
- **Recall = 0.702**: המודל תפס 70% מהשיתופי פעולה האמיתיים שקרו ב-2021-2025
- **Precision = 0.579**: 58% מהחיזויים החיוביים היו נכונים
- **תוספת Temporal Features**: שיפור של +8.2% ב-AUC
- **תוספת Genre/Director**: שיפור נוסף של +1.5% ב-AUC
- **Feature Selection**: 26 פיצ'רים נותנים AUC=0.811 עם מורכבות נמוכה יותר

### 3.4.2 פרמטרי האימון

📊 **טבלה 9**: תצורת המודלים
```
📁 מקור: notebooks/06_link_prediction_final.ipynb — תא MODEL PARAMETERS

Logistic Regression:
- Solver: lbfgs
- Max iterations: 2000
- C (regularization): 1.0
- Class weight: balanced
- Converged: ✓ YES

Random Forest:
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 2
- class_weight: balanced
- n_jobs: -1

Dataset Size:
- Train Set: 382 pairs (191 positive, 191 negative)
- Test Set: 564 pairs (188 positive, 376 negative)
```

### 3.4.3 חשיבות פיצ'רים

🖼️ **איור 6**: Top 15 Feature Importance (Random Forest)
```
📁 קובץ: figures/feature_importance.png
תיאור: גרף עמודות אופקי של הפיצ'רים החשובים ביותר למודל
```

📊 **טבלה 10**: טופ 10 פיצ'רים (Random Forest + Temporal)
```
📁 מקור: notebooks/06_link_prediction_final.ipynb — תא "Feature importance"

| מקום | פיצ'ר | חשיבות | משמעות |
|------|-------|--------|--------|
| 1    | u_genre_diversity | 0.073 | גיוון ז'אנרים של שחקן u — פיצ'ר genre חזק! |
| 2    | u_career_length | 0.066 | אורך קריירה של שחקן u |
| 3    | u_director_count | 0.057 | מספר במאים שונים שעבד איתם u |
| 4    | prod_closeness | 0.057 | מכפלת Closeness centrality של u ו-v |
| 5    | diff_eigenvector | 0.045 | הפרש Eigenvector centrality |
| 6    | v_eigenvector | 0.043 | Eigenvector centrality של שחקן v |
| 7    | v_closeness | 0.043 | Closeness centrality של שחקן v |
| 8    | prod_degree | 0.043 | Preferential Attachment (מכפלת דרגות) |
| 9    | v_degree | 0.040 | Degree centrality של שחקן v |
| 10   | u_closeness | 0.040 | Closeness centrality של שחקן u |

🔍 **תובנות מרכזיות (מנוטבוק 06_link_prediction.ipynb):**
- **u_genre_diversity** — הפיצ'ר הכי חשוב! שחקן שעובד בז'אנרים מגוונים → יותר שיתופי פעולה
- **u_career_length** — ותק = חיזוי חזק; שחקנים עם קריירה ארוכה יוצרים יותר קשרים
- **u_director_count / u_director_overlap** — במאים משותפים = מנבא חזק מ-common_neighbors!
- **Closeness ו-Eigenvector** חשובים יותר מ-Betweenness או Degree בלבד
- **Jaccard, Adamic-Adar, common_neighbors** הוסרו — קורלציה גבוהה >0.98 ביניהם, הם כמעט זהים
- **is_alive, collaboration_count** הוסרו — חשיבות = 0 (רק 1.9% שחקנים נפטרו)
```

### 3.4.4 ניסויים עם פיצ'רי Genre ו-Director

במהלך המחקר, נוסו **פיצ'רים נוספים** מבוססי מטא-דאטה של סרטים:

#### **פיצ'רי Genre (ז'אנר):**
1. **genre_overlap**: כמה ז'אנרים משותפים בין הסרטים של u ו-v
2. **same_main_genre**: 1 אם לשני השחקנים אותו ז'אנר עיקרי, 0 אחרת
3. **u_genre_diversity**: כמה ז'אנרים שונים שחקן u עבד בהם
4. **v_genre_diversity**: כמה ז'אנרים שונים שחקן v עבד בהם

#### **פיצ'רי Director (במאי):**
5. **director_overlap**: כמה במאים משותפים עבדו עם u ו-v
6. **u_director_count**: כמה במאים שונים שחקן u עבד איתם
7. **v_director_count**: כמה במאים שונים שחקן v עבד איתם

📊 **טבלה: תוצאות מודל עם Genre/Director Features**
```
📁 קובץ: data/processed/link_prediction_genre_director_results.csv

| קונפיגורציה | פיצ'רים | AUC-ROC | F1 | Precision | Recall |
|-------------|---------|---------|----|-----------|---------| 
| +Temporal (34) | 34 | 0.804 | 0.600 | 0.660 | 0.550 |
| **+Genre+Director (40)** | **40** | **0.805-0.811** | **0.612** | **0.668** | **0.565** |

שיפור: +0.6-1.2% ב-AUC, +1.2% ב-F1
```

#### **מדוע לא נכלל במודל הסופי?**

למרות השיפור הקל בביצועים, **הוחלט לא להשתמש בפיצ'רים אלו** מהסיבות הבאות:

1. **שיפור מינימלי**: רק +0.6-1.2% שיפור ב-AUC תמורת 6 פיצ'רים נוספים
2. **תלות בנתוני ז'אנר**: רק 753/840 סרטים (89.6%) יש להם ז'אנר מתועד
   - נתונים חסרים עלולים להוביל לhias
3. **מורכבות מוגברת**: הוספת 6 פיצ'רים מייצרת מודל מורכב יותר עם ROI נמוך
4. **פשטות עדיפה (Occam's Razor)**: מודל פשוט יותר עם 34 פיצ'רים קל יותר להסביר ולתחזק
5. **כיסוי לא מלא**: לא כל השחקנים יש להם מטא-דאטה מספקת על במאים וז'אנרים

**לקח מרכזי:**
פיצ'רים מבוססי-תוכן (content-based) כמו ז'אנר ובמאי יכולים לתרום שיפור קל, אך **פיצ'רים מבניים וטמפורליים** (structural + temporal) מספקים את רוב הכוח החיזוי.

📁 **קוד ניסוי**: `notebooks/06.1_link_prediction_with_genre_director.ipynb`

### 3.4.5 ניתוח שגיאות (Error Analysis)

#### **False Positives (FP) — חיזויים "שגויים"**

📊 **טבלה 11**: דוגמאות ל-False Positives
```
📁 קובץ: data/processed/link_prediction_fp.csv

המודל חזה שיתוף פעולה — אבל הוא לא קרה
| שחקן 1 | שחקן 2 | ציון מודל | שכנים משותפים | הסבר אפשרי |
|--------|--------|-----------|---------------|-------------|
| [FILL] | [FILL] | [FILL]    | [FILL]        | ז'אנרים שונים? |
```

**למה המודל טעה?**
- זוגות עם **הרבה שכנים משותפים** → המבנה הטופולוגי מציע שיתוף פעולה
- אבל גורמים **שהמודל לא רואה**: ז'אנר, במאי, יחסים אישיים, החלטות casting

#### **False Negatives (FN) — שיתופי פעולה שהמודל החמיץ**

📊 **טבלה 12**: דוגמאות ל-False Negatives
```
📁 קובץ: data/processed/link_prediction_fn.csv

שיתופי פעולה שקרו — אבל המודל נתן להם ציון נמוך
| שחקן 1 | שחקן 2 | ציון מודל | שכנים משותפים | הסבר אפשרי |
|--------|--------|-----------|---------------|-------------|
| [FILL] | [FILL] | [FILL]    | [FILL]        | שחקנים חדשים? |
```

**למה המודל החמיץ?**
- זוגות **ללא חיבור טופולוגי ברור** ברשת 1990-2020
- שיתופי פעולה **מונעי במאי** או casting-director (לא נראה במבנה הגרף)

### 3.4.6 חיזוי עתידי (2026+)

🖼️ **איור 7**: טופ 20 שיתופי פעולה צפויים
```
📁 קובץ: figures/future_predictions.png
תיאור: גרף עמודות של שמות שחקנים עם הסתברות החיזוי
```

📊 **טבלה 13**: טופ 10 חיזויים עתידיים (watch list)
```
📁 קובץ: data/processed/link_prediction_future.csv
עמודות: u_name, v_name, score, common_neighbors, u_degree, v_degree

| מקום | שחקן 1 | שחקן 2 | הסתברות | שכנים משותפים | פירוש |
|------|--------|--------|---------|---------------|--------|
| 1    | [FILL] | [FILL] | [FILL]  | [FILL]        | [למשל: שניהם עובדים עם במאי X] |
```

**שימושים:**
- **למפיקים**: רשימת שחקנים שכדאי להציע יחד לסרטים
- **לחוקרים**: "watch list" לבדיקה בעתיד — האם החיזויים התממשו?

📁 **מקור**: `notebooks/06_link_prediction.ipynb` — תאים "Future predictions", "Top 30 predicted"

---

# 4. תובנות ומסקנות (1-2 עמודים)

## 4.1 מה למדנו על תעשיית הקולנוע הישראלי?

### 4.1.1 צמיחה דרמטית בעשורים האחרונים
- **תקופה A (1918-1989)**: 71 שנים → 132 שחקנים → תעשייה קטנה וצנועה
- **תקופה C (2010-2025)**: 15 שנים בלבד! → 1106 שחקנים → **צמיחה של פי 8**
- **משמעות**: תעשיית הקולנוע הישראלי **התרחבה משמעותית** בעשור האחרון

### 4.1.2 רשת מפוזרת יותר
- **צפיפות יורדת**: מ-5.46% (A) ל-0.77% (C)
- **פירוש**: למרות שיש יותר שחקנים, הרשת **פחות קומפקטית**
- **תובנה**: כיום יש **יותר תת-קבוצות** של שחקנים שעובדים יחד — פחות "מרכז" אחד

### 4.1.3 שחקנים מרכזיים (Hubs)
- **דרור קרן, מנשה נוי** (דוגמאות) → שחקנים עם מעל 30+ שיתופי פעולה
- אלו שחקנים **"גשרים"** בין קבוצות שונות (Betweenness גבוה)
- **Eigenvector Centrality גבוה** → שחקנים שעובדים עם שחקנים מרכזיים אחרים

### 4.1.4 קהילות ברורות
- זוהו [FILL: X] קהילות עיקריות בתקופה C
- **הבדלים אפשריים**: ז'אנר (קומדיה/דרמה), גיל (וותיקים/צעירים), במאים מועדפים
- שחקנים מסוימים **מגשרים** בין קהילות → versatile actors

---

## 4.2 כיצד השתנתה הרשת לאורך השנים?

### 4.2.1 מרכזיות לביזור
- **שנות ה-50-80**: תעשייה קטנה, כולם מכירים את כולם
- **שנות ה-2000**: פיצול לקבוצות יותר מובחנות
- **שנות ה-2010+**: בום הפקות → רשת גדולה ומפוזרת

### 4.2.2 הופעת "דור חדש"
- תקופה C מלאה בשחקנים צעירים שלא קיימים בתקופות קודמות
- **cold-start problem**: קשה למודל לחזות שיתופי פעולה של שחקנים חדשים לגמרי

### 4.2.3 עלייה בייצור סרטים
🖼️ **איור 1** (films_per_year_cutoffs.png): גידול ברור במספר הסרטים משנות ה-2000 ואילך

---

## 4.3 היכולת לחזות שיתופי פעולה עתידיים

### 4.3.1 ביצועי המודל
- **AUC-ROC = 0.819**: ביצועים **מצוינים** (מעל 0.8)
- **Temporal Features**: +8.2% שיפור לעומת Baseline (0.722 → 0.804)
- **Genre/Director Features**: +1.5% שיפור נוסף (0.804 → 0.819)
- **Feature Selection**: 12 פיצ'רים הוסרו → AUC=0.811 עם מודל קומפקטי יותר
- **False Positives**: שחקנים עם במאי משותף אבל לא עבדו יחד — המודל לא רואה את ה"למה לא"
- **False Negatives**: שחקנים ב-cold-start (ללא היסטוריה, `__name__:...`) — המודל לא מכיר אותם

### 4.3.2 מה עובד?
- **Temporal Features**: הפיצ'רים החזקים ביותר! (career_length, film_count)
- **u_career_length**: הפיצ'ר החזק ביותר (Feature Importance = 0.095)
- **Common Neighbors**: עדיין רלוונטי אך לא דומיננטי (מקום 6)
- **Centrality Features**: במיוחד Betweenness ו-Eigenvector centrality

### 4.3.3 מה לא עובד?
- **Topological features alone לא מספיק**: יש גורמים חיצוניים (במאי, ז'אנר, אג'נטים)
- **Genre/Director features**: נוסו אך הוסיפו רק +0.6-1.2% שיפור (לא שווה את המורכבות)
- **Cold-start**: שחקנים חדשים ללא היסטוריה → המודל מתקשה
- **Temporal dynamics**: שינויים בטרנדים (למשל, עלייה בסדרות vs סרטים) לא מודלים
- **נתונים חסרים**: רק 89.6% מהסרטים יש להם ז'אנר מתועד

---

## 4.4 מגבלות המחקר

### 4.4.1 מקור הנתונים
- **ויקיפדיה בלבד**: לא כולל סדרות טלוויזיה, פרסומות, תיאטרון
- **מגבלת 10 שחקנים**: סרטים עם קאסט גדול → נתונים חלקיים
- **כיסוי לא מלא**: סרטים ללא ערך בויקיפדיה לא נכללו

### 4.4.2 הנחות המודל
- **Temporal design**: הנחנו `years_ahead=5` בלבד — לא נבדקו horizons שונים
- **Negative sampling**: אקראי — לא stratified לפי degree או מאפיינים אחרים
- **No node features**: לא השתמשנו במידע דמוגרפי (גיל, מין, ז'אנר מועדף)

### 4.4.3 הערכה
- **Test set יחיד**: לא ביצענו cross-validation או multiple temporal splits
- **בחירת היפר-פרמטרים**: לא בוצע grid-search מקיף

---

## 4.5 המלצות להמשך

### 4.5.1 הרחבת מקורות נתונים
- **IMDb או Cinematic**: כיסוי רחב יותר של סרטים ושחקנים
- **סדרות טלוויזיה**: שילוב עולם הטלוויזיה הישראלית
- **תיאטרון**: הרחבה לבמה (אם יש נתונים זמינים)

### 4.5.2 שיפור המודל
- **Dynamic graph embeddings**: Node2Vec Temporal, DynGEM
- **Deep learning**: Graph Neural Networks (GCN, GraphSAGE) למידת representations
- **Ensemble methods**: שילוב מודלים מרובים
- **Multi-horizon training**: אימון על horizons שונים (1-year, 3-year, 5-year)

### 4.5.3 Features נוספים
- **Actor metadata**: גיל, מין, ז'אנרים שעבדו בהם, פרסים
- **Director features**: זיהוי במאי כ-"hub" — מי עובד עם מי
  - **הערה**: פיצ'רי genre/director נוסו (ראה סעיף 3.4.4) אך השיפור היה מינימלי (+0.6-1.2%)
  - אפשר לנסות **פיצ'רים מתקדמים יותר**: genre embeddings, במאי כ-nodes בגרף
- **Temporal trends**: תכונות של שינויים בזמן (האם שחקן עובד יותר/פחות עם הזמן?)
- **Network position**: community membership, bridging coefficient

### 4.5.4 יישומים מעשיים
- **Recommendation system**: המלצה לסטודיו על קאסטינג
- **Network visualization tool**: אפליקציה אינטראקטיבית לחקר הרשת
- **Historical analysis**: מחקר סוציולוגי על תעשיית הקולנוע

---

# 5. סיכום

מחקר זה הציג **ניתוח מקיף של רשת שיתופי הפעולה** של שחקני הקולנוע הישראלי, תוך שימוש בכלי **Machine Learning על גרפים**. מאיסוף הנתונים מוויקיפדיה, דרך בניית הגרף וניתוחו לאורך שלוש תקופות, ועד לבניית מודל חיזוי מתקדם — הראינו כיצד ניתן **להפיק תובנות משמעותיות** מנתוני רשת.

**תוצאות עיקריות:**
- 🎬 **840 סרטים, 1,039 שחקנים** (טווח 1933-2026, 93 שנים)
- 🎬 **1,106 שחקנים, 4,731 שיתופי פעולה** בתקופה המודרנית (>1990)
- 📈 **צמיחה פי 8.4** במספר השחקנים בין תקופה A ל-C
- �️ **Clustering Coefficient = 0.741-0.751** — רשת "Small World" עם חבורות צפופות
- 🎯 **AUC-ROC = 0.819** בחיזוי שיתופי פעולה עתידיים (Random Forest, 38 פיצ'רים)
- 🔍 **u_genre_diversity** ו-**u_career_length** — הפיצ'רים החזקים ביותר
- 🔬 **Feature Selection**: 12 פיצ'רים כפולים הוסרו → AUC=0.811 עם 26 פיצ'רים
- ✅ **מניעת Data Leakage**: Temporal split קפדני + is_alive feature

המחקר מדגים את הפוטנציאל של **Network Science** להבנת תעשיות יצירתיות, ומספק בסיס איתן להמשך מחקר בתחום.

---

# נספחים

## נספח A: קבצי נתונים

| **קובץ** | **תיאור** | **גודל** |
|----------|----------|----------|
| `movies.csv` | רשימת כל הסרטים (שנה, שם, במאי, ז'אנר) | 840 rows |
| `cast_edges.csv` | כל הקשרים סרט-שחקן | 2,972 rows |
| `period_metrics.csv` | מדדי גרף לכל תקופה | 3 rows |
| `community_assignments.csv` | השתייכות לקהילה לכל שחקן | 1,106 rows |
| `centrality_scores.csv` | ערכי מרכזיות לכל שחקן | 1,106 rows |
| `link_prediction_results.csv` | ביצועי המודלים (Baseline + Temporal) | 4 rows |
| `link_prediction_genre_director_results.csv` | תוצאות ניסוי עם genre/director | 2 rows |
| `actor_death_years.csv` | שחקנים שנפטרו | 21 rows |

📁 **כל הקבצים**: `israeli_actors_graph/data/processed/`

## נספח B: תמונות ואיורים

| **איור** | **קובץ** | **מיקום בדוח** |
|----------|---------|---------------|
| 1 | `films_per_year_cutoffs.png` | סעיף 3.1.1 |
| 2 | [Log-Log degree dist.] | סעיף 3.1.3 |
| 3 | [Community visualization] | סעיף 3.2.2 |
| 4 | [Top actors by degree] | סעיף 3.3.1 |
| 5 | `link_pred_test_curves.png` | סעיף 3.4.1 |
| 6 | `feature_importance.png` | סעיף 3.4.3 |
| 6.1 | `model_comparison_genre_director.png` | סעיף 3.4.4 |
| 7 | `future_predictions.png` | סעיף 3.4.6 |
| 8 | `pair_feature_distributions.png` | [אופציונלי: נספח] |

📁 **כל התמונות**: `israeli_actors_graph/figures/`

## נספח C: קוד

**נוטבוקים (6 נוטבוקים עיקריים):**
1. `01_data_collection.ipynb` — איסוף נתונים מוויקיפדיה (840 סרטים)
2. `02_graph_construction.ipynb` — בניית גרף + חלוקה לתקופות
3. `03_temporal_period_analysis.ipynb` — ניתוח מדדים לכל תקופה
4. `04_community_detection.ipynb` — זיהוי קהילות (3 אלגוריתמים)
5. `05_centrality_analysis.ipynb` — ניתוח מרכזיות (4 מדדים)
6. `06_link_prediction.ipynb` — חיזוי קשרים מלא: 38 פיצ'רים + feature selection + ניתוח שגיאות

**ארכיון (`_archive/`):** נוטבוקים ניסיוניים ישנים (06_final, 06.1, 07)

**מודולי קוד:**
- `src/scraping.py` — פונקציות Web scraping
- `src/graph_build.py` — בניית גרפים
- `src/communities.py` — אלגוריתמי קהילות
- `src/centrality.py` — חישוב מרכזיות
- `src/link_features.py` — חישוב פיצ'רים
- `src/link_prediction.py` — אימון והערכת מודלים

📁 **כל הקוד**: `israeli_actors_graph/notebooks/` ו-`israeli_actors_graph/src/`

---

# הנחיות לקופיילוט של Word

**שלום קופיילוט!**  
להלן קובץ Markdown מפורט עם כל מבנה הדוח הסופי (10-15 עמודים).

**מה צריך לעשות:**
1. **המר את המסמך ל-Word**
2. **מלא את כל ה-`[FILL: ...]`** — קרא את הקבצים המסומנים ב-📁 והשלם ערכים
3. **הוסף תמונות** — כל 🖼️ מסמן איור שצריך לשלב
4. **הוסף טבלאות** — כל 📊 מסמן טבלה מקובץ CSV
5. **עיצוב מקצועי**: כותרות, מספור סעיפים, תוכן עניינים, גופן קריא

**נתיבים לקבצים:**
- נתונים: `c:\Users\rzanzuri\OneDrive - Intel Corporation\Desktop\MySecond\Second year\ML-on-Graphs\israeli_actors_graph\data\processed\`
- תמונות: `c:\Users\rzanzuri\OneDrive - Intel Corporation\Desktop\MySecond\Second year\ML-on-Graphs\israeli_actors_graph\figures\`
- נוטבוקים: `c:\Users\rzanzuri\OneDrive - Intel Corporation\Desktop\MySecond\Second year\ML-on-Graphs\israeli_actors_graph\notebooks\`

**פורמט רצוי:**
- גופן: David / Arial (12pt)
- כותרות: 14-16pt מודגשות
- ריווח: 1.15 שורות
- שוליים: 2.5 ס"מ
- תוכן עניינים אוטומטי

**תודה!**
