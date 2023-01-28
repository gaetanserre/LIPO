## Numpy parser

Checks if a string is a valid numpy function expression.

### Usage
```bash
cd build
make
./numpy_parser "<numpy expression>"
```

### Example
```bash
./numpy_parser "np.sin(x[0]**2) - np.cos(x[1])"
```