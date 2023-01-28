default: all

MAIN := numpy_parser

all:
	dune build $(MAIN) \
	&& cp -f _build/default/$(MAIN)/$(MAIN).exe ../$(MAIN).exe \
	&& dune clean
clean:
	rm -rf $(MAIN) && cd ../ && dune clean