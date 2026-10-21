.PHONY: help install test data features baselines train tables reproduce run clean

VENV ?= $(HOME)/venvs/mc
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

help:
	@echo "Memory Compiler - Target Commands"
	@echo "  make install     Install dependencies in virtual environment"
	@echo "  make test        Execute test suite (pytest)"
	@echo "  make data        Download or prepare dataset"
	@echo "  make features    Extract video frame embeddings"
	@echo "  make baselines   Run B0-B2 baseline benchmarks"
	@echo "  make train       Train temporal retrieval network (B3)"
	@echo "  make tables      Generate 5 quantitative result tables"
	@echo "  make reproduce   Execute full reproduction pipeline"
	@echo "  make run         Launch backend API and frontend app"
	@echo "  make clean       Remove cached runs and temporary data"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

data:
	bash scripts/download_data.sh

features:
	$(PYTHON) scripts/extract_features.py --config configs/data/charades_sta.yaml

baselines:
	$(PYTHON) scripts/run_baselines.py --config configs/eval/default.yaml

train:
	$(PYTHON) -m mc.train.trainer --config configs/train/default.yaml

tables:
	$(PYTHON) scripts/make_tables.py --runs_dir runs/ --output_dir results/

reproduce:
	bash scripts/reproduce.sh

run:
	$(PYTHON) -m mc.api.main & cd app && npm run dev

clean:
	rm -rf __pycache__ .pytest_cache mc/**/__pycache__
