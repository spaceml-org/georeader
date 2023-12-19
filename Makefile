build-package:
	rm -rf build/
	rm -rf dist/
	python setup.py sdist bdist_wheel

upload-pip:
	python -m twine upload dist/*

