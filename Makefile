.PHONY: test

test:
	nosetests -v -s gemelli --with-coverage --cover-package=gemelli
