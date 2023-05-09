default: all

MAIN := numpy_parser

all:
	cd src \
	&& ocamlbuild $(MAIN).native \
	&& mv _build/$(MAIN).native ../$(MAIN) \
	&& rm -rf _build $(MAIN).native
clean:
	rm -rf $(MAIN) && cd src/ && rm -rf _build $(MAIN).native