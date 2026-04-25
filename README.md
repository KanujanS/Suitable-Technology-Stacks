# Suitable-Technology-Stacks

GitHub Tech Stack Collector for building a dataset of project characteristics and inferred technology stacks from GitHub repositories.

## Overview

The notebook in this repository searches GitHub repositories across several application domains and collects metadata for each project. It then uses repository descriptions, topics, READMEs, and language data to infer likely frontend, backend, and database technologies, along with deployment style, project size, and approximate delivery constraints.

The current notebook target is 1000 projects, and it writes the results to an Excel file named `github_projects_1000.xlsx`.

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
6. Saves the final dataset to an Excel workbook.

## Requirements

- Python 3
- `requests`
- `pandas`
- A GitHub personal access token

## Setup

1. Open `github_collector_1000.ipynb`.
2. Replace `YOUR_GITHUB_TOKEN` with your GitHub token.
3. Run the notebook cells from top to bottom.

## Output

When the notebook finishes, it saves the dataset to `github_projects_1000.xlsx`. If run in Google Colab, the file is also downloaded automatically.

## Notes

- The notebook includes a built-in delay between requests to reduce API pressure.
- It also checks GitHub rate limits periodically during long runs.
