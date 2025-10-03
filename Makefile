.PHONY: all data kdef tess

all: data

data: kdef tess

kdef:
	python -m src.emotional_n_back.main kdef

tess:
	python -m src.emotional_n_back.main tess
