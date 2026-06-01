# Supply-Chain Disruption Database

This repository contains a data pipeline for identifying, extracting, and validating supply-chain disruption events from global news reporting. The goal is to produce structured event data and quantitative indicators for downstream disruption-risk and criticality analysis.

## Overview

The pipeline ingests global news data, filters for disruption events, extracts these using a large language model (LLM), and validates these events against external reference datasets. The resulting database in conjuction with supplementation of missing indicators will in later stages allow for disruption-likelihood modelling.

## Key Features

* Daily ingestion of global news via GDELT
* Supervised filtering of irrelevant articles
* LLM-based extraction of structured disruption events
* Coverage of physical and socio-political disruption types
* Modular validation against external datasets (e.g. ACLED, MMAD, ICEWS) (WIP)

## Project Structure

* `pipeline.py`  -  news ingestion, filtering, and relevance classification
* `DatabaseBuilder/`  -  LLM-based disruption extraction
* `DatabaseValidation/`  -  validation against external datasets
* `data/`  -  intermediate and output data files

## Installation

* Python 3.10+
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Configure required API keys in a `.env` file

## Usage

1. Run `pipeline.py` with a target date to retrieve and filter news articles
2. Run `DatabaseBuilder/DisruptionExtractor.py` to extract structured events
3. (Optional) Use display scripts to inspect extraction outputs
4. Validation scripts are located in `DatabaseValidation/` (at the moment these are simply just smoke tests)


## Current Limitations

* Validation is incomplete
* Indicator retrieval and likelihood modelling are still in progress
* The relevance classifier requires additional labelled data to reduce overheads downstream

## Roadmap

* Complete validation across all disruption categories
* Finalise indicator population and missing-data handling
* Implement statistical and machine-learning models for disruption likelihood estimation

## License

This project is intended for academic and research use. External datasets are subject to their own licensing terms.
