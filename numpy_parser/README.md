## Numpy parser

Checks if a string is a valid Numpy function expression given a list of Numpy primitives. It does not check if the expression is valid for a given set of arguments, only if it is a valid Numpy expression in terms of grammar. Returns 0 if the expression is valid, an error otherwise. The purpose of this program is to be used within another program, e.g. to check if a user input is a not a dangerous expression.

### Build
Requirements:
- ocaml compiler
- ocamllex
- menhir
- ocamlbuild

```bash
make
```

### Usage
```bash
make
./numpy_parser numpy_primitives.txt "<numpy expression>"
```

### Examples
```bash
./numpy_parser numpy_primitives.txt "np.sin(x[0]**2) - np.cos(x[1])"
```
```bash
./numpy_parser numpy_primitives.txt "os.remove('\')"
```