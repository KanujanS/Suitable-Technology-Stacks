# Suitable-Technology-Stacks

GitHub Tech Stack Collector for building a dataset of project characteristics and inferred technology stacks from GitHub repositories.

## Overview

The notebook in this repository, `github_collector.ipynb`, searches GitHub repositories across several application domains and collects metadata for each project. It uses repository descriptions, topics, READMEs, and language statistics to infer likely frontend, backend, and database technologies, along with deployment style, project size, and approximate delivery constraints.

By default the notebook defines ~75 curated search queries and `MAX_PAGES = 3`, so the maximum candidate rows before deduplication is roughly 75 * 100 * 3 = 22,500 (actual saved rows will be lower after deduplication). The default output file is `github_projects_data.xlsx`.

## What It Collects

Each collected record includes:

- Project name and GitHub URL
- Domain and short project description
- Functional and non-functional requirements
- Project size, team size, budget level, and duration estimate
- Deployment type
- Frontend, backend, and database technology suggestions
- GitHub stars, forks, and primary language

## How It Works

The notebook:

1. Queries the GitHub Search API with a curated set of search strings per domain.
2. Fetches the repository README and language breakdown when needed.
3. Detects technologies using keyword matching and language-to-stack fallbacks.
4. Classifies repository size from stars and forks.
5. Fills in missing project attributes using domain and size heuristics.
6. Saves the final dataset to an Excel workbook (default: `github_projects_data.xlsx`).

## Requirements

- Python 3
- `requests`
- `pandas`
- A GitHub personal access token

## Setup

1. Open `github_collector.ipynb`.
2. Replace the `GITHUB_TOKEN` value in the notebook with your GitHub personal access token.
3. Optionally adjust `OUTPUT_FILE`, `MAX_PAGES`, and delay constants near the top of the notebook.
4. Run the notebook cells from top to bottom (or execute the notebook in Google Colab).

Alternative: convert to a script and run from the command line:

```
jupyter nbconvert --to script github_collector.ipynb
python github_collector.py
```

## Output

When the notebook finishes, it saves the dataset to `github_projects_data.xlsx` by default. If run in Google Colab, the file is also downloaded automatically.

## Notes

 - The notebook includes a built-in delay between requests to reduce API pressure.
 - It checks GitHub rate limits periodically during long runs and pauses if limits are low.
