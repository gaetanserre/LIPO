default: all

MAIN := numpy_parser

all:
	cd ../ \
	&& dune build $(MAIN) \
	&& cp -f _build/default/$(MAIN)/$(MAIN).exe build/$(MAIN) \
	&& dune clean \
	&& cd build

clean:
	rm -rf $(MAIN) && cd ../ && dune clean