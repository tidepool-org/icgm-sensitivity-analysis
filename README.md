
# iCGM Sensitivity Analysis
#### -- **Project Status**: Active

## Project Objective
The purpose of this project is to assess how the Tidepool Loop Algorithm is affected by the uncertainties/inaccuracies in continuous glucose monitor (CGM) data as defined by the iCGM special controls.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* Python
* Pandas, Google Colab

## Project Description
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting

## Getting Started

Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

**Setup Virtual Environment** 

The iCGM Sensitivity Analysis is run within an anaconda virtual environment. [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) can be used to install the `conda` command line tool.

1. In a teriminal, navigate to this repository's root folder where the `environmental.yml` file is located.
2. Run `conda env create`. This will download all of the pipeline's package dependencies and install them in a virtual environment named **isa** (**i**CGM **s**ensitivity **a**nalysis)
3. Run `conda activate isa` to activate the environment and `conda deactivate` at anytime to exit.

**Pipeline Execution**

The entire processing pipeline and risk simulator are run from the src directory in order:

```
python batch-icgm-condition-stats.py
python snapshot-processor.py
python risk-simulation-pipeline.py
```

Sample data necessary to run all parts of the simulationis has been added [here](./src) within this repo.

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


## Contributing Members

| Name                                           | Slack Handle     |
| ---------------------------------------------- | ---------------- |
| [Ed Nykaza](https://github.com/ed-nykaza)      | @ed              |
| [Jason Meno](https://github.com/jameno)        | @jason           |
| [Cameron Summers](https://github.com/scaubrey) | @Cameron Summers |
