install:
	cd ./graphsampler
	pip install -e graphsampler
	python ./graphsampler/demo.py

run: 
	python ./graphsampler/main.py

clean: 
	\rm -rf ./graphsampler/results*

all.tar:
	tar cvfh all.tar *