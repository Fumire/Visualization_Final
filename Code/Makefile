all: python

python: standalone.py
	python3 $^
.PHONY += python

clean: clean_figure clean_pickle clean_csv
.PHONY += clean

clean_pickle:
	rm -fv .*.pkl
.PHONY += clean_pickle

clean_figure:
	rm -fv figures/*
.PHONY += clean_figure

clean_csv:
	rm -fv csv/*
.PHONY += clean_csv
