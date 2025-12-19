## Project Overview

This is a data science project analyzing League of Legends champion balance changes (buffs and nerfs) across Season 11. The project uses machine learning to predict which champions are likely to receive balance changes based on their performance statistics across different rank tiers.

**Project Goal**: Predict champion buffs/nerfs using historical performance data (win rate, pick rate, ban rate) across different skill levels.

**Key Achievement**: Decision Tree classifier achieving 82.6% accuracy in predicting balance changes.

## Repository Structure

The project follows standard data science best practices with clear separation between raw data, processed data, analysis notebooks, and outputs.

```
buff_project/
├── data/                                    # All data files
│   ├── raw/                                 # Immutable original data (never modify)
│   │   ├── season_11/                       # Season 11 raw data by patch
│   │   │   ├── Challenger/                  # 11.1-11.17 challenger tier CSVs
│   │   │   ├── irontogold/                  # 11.1-11.19 low rank CSVs
│   │   │   ├── plattogm/                    # Platinum to Grandmaster CSVs
│   │   │   ├── changes/                     # 11.1-11.19 patch change CSVs
│   │   │   ├── latestpatch/                 # Most recent patch data
│   │   │   └── combined/                    # Alternative combined datasets
│   │   └── historic/                        # Legacy data (Patches 8.24, 9.24, 10.25)
│   │       ├── [patch]challenger.csv
│   │       ├── [patch]changes.csv
│   │       └── [patch][rank-range].csv
│   ├── processed/                           # Cleaned and merged datasets
│   │   ├── S11combined.csv                  # Combined Season 11 (all ranks)
│   │   ├── S11challenger.csv                # Challenger tier only
│   │   ├── S11irontogm.csv                  # Iron to Grandmaster
│   │   ├── S11buffandnerf.csv               # Labeled dataset for ML training
│   │   ├── 11.18combined.csv                # Patch 11.18 specific combined
│   │   └── allrankseason11.csv              # All ranks aggregated
│   └── reference/                           # Reference and lookup files
│       └── champs.csv                       # Master list of all League champions
│
├── notebooks/                               # Jupyter notebooks organized by pipeline stage
│   ├── 01_data_collection/                  # Web scraping and data acquisition
│   │   ├── champ_historic_data_webscraping.ipynb
│   │   └── patch_history_webscraping.ipynb
│   ├── 02_data_cleaning/                    # Data preprocessing and merging
│   │   └── data_cleaning.ipynb
│   └── 03_modeling/                         # Model training and evaluation
│       ├── decision_tree_modeling.ipynb     # Primary model (82.6% accuracy)
│       ├── SVM model.ipynb                  # Alternative SVM classifier
│       └── Untitled.ipynb                   # Experimental/scratch notebook
│
├── models/                                  # Trained models and model artifacts
│
├── reports/                                 # Generated analysis and documentation
│   └── figures/                             # Visualizations and plots
│       └── bufftree.png                     # Decision tree visualization (971KB)
│
├── README.md                                # Project documentation

```

## Data Pipeline & Workflow

### 1. Data Collection (Web Scraping)

**File**: `notebooks/01_data_collection/champ_historic_data_webscraping.ipynb`

- **Source**: metasrc.com
- **Function**: `getdataframe(patch)` - Scrapes champion statistics for a given patch
- **Rank Tiers Scraped**:
  - Iron/Bronze/Silver/Gold (low tier)
  - Platinum/Diamond/Master/Grandmaster (high tier)
  - Challenger (elite tier)
- **Data Collected**: Champion name, role, win rate, ban rate, pick rate
- **Output Format**: Separate CSVs for each rank tier and patch
- **Note**: Champion names are duplicated in HTML and need to be split (uses `champ[len(champ)/2:]`)

**File**: `notebooks/01_data_collection/patch_history_webscraping.ipynb`

- **Source**: pcgamesn.com
- **Function**: `getchanges(patch)` - Identifies buffs/nerfs from patch notes
- **Change Types**:
  - `buff` - Champion received buffs
  - `nerf` - Champion received nerfs
  - `tweak` - Champion changed but not explicitly buffed/nerfed
  - `no change` - Champion unchanged
- **Uses**: `data/reference/champs.csv` as reference list to check against patch notes

### 2. Data Cleaning & Merging

**File**: `notebooks/02_data_cleaning/data_cleaning.ipynb`

- **Function**: `dataclean(patch)` - Merges champion stats with patch change data
- **Process**:
  1. Load three rank-tier CSVs for a patch
  2. Load next patch's changes (to see what happened AFTER these stats)
  3. Merge stats with change labels using champion name
  4. Output labeled datasets
- **Function**: `dataclean2(patch)` - Combines all three rank tiers into single dataset
- **Key Logic**: Stats from patch N are labeled with changes from patch N+1 (predictive modeling)

### 3. Model Training

**File**: `notebooks/03_modeling/decision_tree_modeling.ipynb` (Primary Model)

- **Algorithm**: Decision Tree Classifier (sklearn)
- **Features**:
  - `winrate` - Champion win percentage
  - `rank` - Encoded rank tier (0=irontogold, 1=plattogm)
- **Target**: `change` - Buff/nerf/no change classification
- **Hyperparameters**:
  - `criterion="entropy"` - Information gain splitting
  - `max_depth=8` - Prevents overfitting
- **Performance**: 82.6% accuracy on test set
- **Train/Test Split**: 70/30, random_state=3
- **Visualization**: Exports decision tree to `reports/figures/bufftree.png`

**File**: `notebooks/03_modeling/SVM model.ipynb` (Alternative Model)

- **Algorithm**: Support Vector Machine with RBF kernel
- **Features**: `winrate`, `rank`, `banrate`, `pickrate` (more comprehensive)
- **Target**: Binary classification (buff=2, nerf=4)
- **Performance**:
  - F1-score: 0.7284
  - Jaccard score: 0.6864
- **Observation**: Better at predicting buffs than nerfs

## Data Schema

### Champion Statistics CSV Format

```csv
champ,role,winrate,banrate,pickrate,change
Aatrox,TOP,48.17,4.48,4.68,no change
```

**Columns**:
- `champ`: Champion name (string)
- `role`: TOP, JUNGLE, MID, ADC, SUPPORT
- `winrate`: Win percentage as float (e.g., 48.17 = 48.17%)
- `banrate`: Ban percentage as float
- `pickrate`: Pick percentage as float
- `rank`: Rank tier (irontogold, plattogm, challenger) - added during merging
- `change`: buff, nerf, tweak, no change - label for ML

### Patch Naming Convention

- Format: `[season].[patch_number]` (e.g., `11.18` = Season 11, Patch 18)
- Range in this project: Patches 11.1 through 11.19

## Technologies Used

- **Python 3.x**
- **Data Collection**: requests, BeautifulSoup (html5lib parser)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
  - DecisionTreeClassifier
  - SVM (Support Vector Machines)
  - LabelEncoder for categorical features
- **Visualization**: matplotlib, pydotplus (decision tree graphs)
- **Development**: Jupyter Notebook

## Development Conventions

### Code Style

1. **Function Naming**: Lowercase with no underscores (e.g., `getdataframe`, `getchanges`)
2. **DataFrame Operations**: Uses deprecated `df.append()` - consider updating to `pd.concat()`
3. **File Naming**: `[patch][datatype].csv` pattern (e.g., `11.18challenger.csv`)
4. **String Replacement**: Uses regex replace to clean percentage signs from scraped data

### Data Quality Considerations

1. **Champion Name Parsing**: Web scraping duplicates champion names in HTML; code splits at midpoint
2. **Rate Limiting**: pcgamesn.com may return "too many requests" error - add delays if scraping bulk data
3. **Missing Data**: Some champion/role combinations may not exist in certain rank tiers
4. **Label Alignment**: Patch N stats are labeled with patch N+1 changes (intentional for prediction)

### Known Issues & Model Limitations

From `SVM model.ipynb` analysis (cell 12 markdown):

> "The model is good at predicting when a champ is likely to be buffed - winrate, playrate and banrate are good indicators. But nerfing is more complex."

**Why Nerf Prediction is Harder**:
1. **Complexity**: Some champions are kept at low win rates intentionally (high skill ceiling)
2. **Pro Play Impact**: Champions strong in professional play get nerfed despite average solo queue stats
3. **Role-Specific Nerfs**: A champion might be overpowered in one role but balanced overall
4. **Mastery Curves**: Win rate among champion experts differs from general population

**Suggested Improvements**:
- Add professional play statistics
- Include win rate among champion masters (high games played)
- Track win rates across all five roles separately
- Consider pick/ban rates in professional matches

## Common Tasks for AI Assistants

### Running the Full Data Pipeline

```python
# 1. Scrape champion data for a patch
exec(open('notebooks/01_data_collection/champ_historic_data_webscraping.ipynb').read())
getdataframe('11.19')  # Creates irontogold, plattogm, challenger CSVs

# 2. Scrape patch changes
exec(open('notebooks/01_data_collection/patch_history_webscraping.ipynb').read())
getchanges('1119')  # Creates changes CSV

# 3. Clean and merge data
exec(open('notebooks/02_data_cleaning/data_cleaning.ipynb').read())
dataclean(19)  # Merges stats with changes from patch 20
```

### Training a New Model

```python
# Decision Tree approach
from decision_tree_modeling import *
# Data is already split and model trained in notebook
# Accuracy available via: metrics.accuracy_score(y_testset, predTree)

# SVM approach
from SVM_model import *
# Uses more features (winrate, rank, banrate, pickrate)
```

### Adding New Champion Data

1. Add champion name to `data/reference/champs.csv` (one name per line)
2. Re-run web scraping functions to collect stats
3. Re-run data cleaning to generate merged datasets
4. Retrain models with updated data

### Updating to New Season

1. Update URL patterns in scraping notebooks (change season number)
2. Create new directory: `data/raw/season_[N]/`
3. Update file paths in cleaning notebooks
4. Verify champion list is current (new champions released)

## Important Notes for AI Assistants

1. **README.md is Empty**: Project documentation should be added there for users
2. **Deprecated pandas Methods**: Code uses `df.append()` which is deprecated in pandas 2.0+
3. **Web Scraping Reliability**: URLs and HTML structure may change; verify sources are still accessible
4. **Rate Limits**: Be respectful of web scraping sources; add delays between requests
5. **Data Freshness**: Project focuses on Season 11 (2021); League of Legends is actively updated
6. **Checkpoint Files**: `.ipynb_checkpoints/` contains auto-saved notebook versions
7. **Image Files**: `reports/figures/bufftree.png` is a large (971KB) decision tree visualization

## Git Workflow

- **Branch Naming**: Use `claude/` prefix for AI-generated branches
- **Commit Style**: Project has minimal commit history; use descriptive messages
- **Recent Commits**: "added some commentary to model after 4 years!" suggests dormant project

## Testing & Validation

- Models use `random_state=3` for reproducibility
- Test set size: 30% of data (decision tree), 20% (SVM)
- No unit tests present; validation is done through notebook outputs
- Confusion matrices available in SVM notebook for detailed performance analysis

## Future Enhancements

Based on analysis in the notebooks, consider:

1. **Feature Engineering**: Add professional play statistics, champion mastery curves
2. **Multi-Class Classification**: Better handling of buff/nerf/tweak/no change (currently binary)
3. **Time Series Analysis**: Predict multiple patches ahead
4. **API Integration**: Use Riot Games API instead of web scraping for more reliable data
5. **Role-Specific Models**: Separate models for each role (TOP, JUNGLE, etc.)
6. **Automated Pipeline**: Schedule regular data collection and model retraining
7. **Web Dashboard**: Visualize predictions and model confidence

## Quick Reference: File Locations

- **Master champion list**: `data/reference/champs.csv`
- **Latest combined dataset**: `data/processed/S11combined.csv`
- **Primary model notebook**: `notebooks/03_modeling/decision_tree_modeling.ipynb`
- **Raw Season 11 data**: `data/raw/season_11/`
- **Model visualization**: `reports/figures/bufftree.png`
- **Training data for SVM**: `data/processed/S11buffandnerf.csv`

---

**Last Updated**: 2025-12-10
**Project Status**: Research/Analysis (dormant since ~4 years ago based on commit messages)
**Contact**: See git commit history for original author
