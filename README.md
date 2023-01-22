## LIPO implementation and demo

### Usage
You need to create a class for your function to maximize. This class must be named `Function` and follow the following interface:
```python
import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = bounds # (min, max) tuple for each dimension (numpy array)
    self.k = k # Lipschitz constant (float)
    pass

  def __call__(self, x: np.ndarray) -> float:
    # Closed form of the function to maximize
    pass
```

Several examples are available in the `functions` folder.

Then, you can run the algorithm with the following command:
```bash
python main.py --function <path_to_your_function_class> -n <number_of_function_eval>
```
Some optional arguments are available, you can see them with the `--help` flag.