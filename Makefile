.PHONY: run smoke smoke-remote report

run:
	git lfs pull
	cd data && unzip -o turingAI_forecasting_challenge_dataset.csv.zip
	uv run python submission/generate_forecasts.py

smoke:
	uv run python submission/aggregator_check.py --local --skip-mse

smoke-remote:
	@test -n "$(OWNER)" || (echo "Usage: make smoke-remote OWNER=<gh-owner> REPO=<gh-repo>"; exit 1)
	@test -n "$(REPO)"  || (echo "Usage: make smoke-remote OWNER=<gh-owner> REPO=<gh-repo>"; exit 1)
	uv run python submission/aggregator_check.py $(OWNER) $(REPO) --skip-mse

report:
	cd submission && pandoc report.md -o report.pdf \
		--pdf-engine=pdflatex \
		--from markdown+raw_tex \
		--highlight-style=tango
