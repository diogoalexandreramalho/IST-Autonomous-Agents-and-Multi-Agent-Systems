.PHONY: all clean local test release
all:
	echo "#!/bin/sh\npython3.8 exercise.py\n" > exercise
	chmod 755 exercise

clean:
	rm -f exercise 86407_delivery.zip
	find tests -name '*_hyp' -delete
	find tests -name '*_err.log' -delete

# Because my version of python3 is python3.8
local: exercise.py
	echo "#!/bin/sh\n/usr/bin/python3 exercise.py\n" > exercise
	chmod 755 exercise

test: local tests/
	./tests/test.sh

release:
	zip 86407_delivery.zip Makefile exercise.py
