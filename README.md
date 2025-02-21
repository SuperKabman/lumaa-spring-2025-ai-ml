# Simple Movie Recommendation System

A content-based movie recommendation system that suggests movies based on text descriptions of what you're looking for. The system uses TF-IDF vectorization and cosine similarity to match your preferences with movie descriptions and genres.

## Dataset

This project uses the TMDB 5000 Movie Dataset, which includes movie overviews, genres, and keywords. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data?select=tmdb_5000_movies.csv) and place `tmdb_5000_movies.csv` in the project root directory.

## Setup

### Requirements
- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the System

Run the recommendation system using:
```bash
python movieRecommender.py
```

The system will prompt you to enter a description of the kind of movie you're looking for. Type 'quit' to exit.

### Example

Input:
```
"I love exciting action movies with lots of adventure"
```

Output:
```
Detected genres from your description: action, adventure

Top Recommendations for You:
-----------------------------

1. Indiana Jones and the Raiders of the Lost Ark
   Match Score: 0.856
   Overview: Archaeologist and adventurer Indiana Jones must retrieve a legendary artifact...

2. Mission: Impossible - Rogue Nation
   Match Score: 0.823
   Overview: A thrilling action-packed spy adventure with death-defying stunts...

[Additional recommendations...]
```

The system extracts relevant genres from your description and combines text similarity with genre matching to provide personalized movie recommendations.

## Salary Expectation

$3000-4000 per month