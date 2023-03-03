## LIPO, AdaLIPO, LIPO-E, and AdaLIPO-E implementation

This repository contains an implementation of the following algorithms:
- [LIPO](https://arxiv.org/abs/2006.04779)
- [AdaLIPO](https://arxiv.org/abs/2006.04779)
- [LIPO-E](https://arxiv.org/abs/2102.02248)
- [AdaLIPO-E](https://arxiv.org/abs/2102.02248)

LIPO-E and AdaLIPO-E are empirical enhancements introduced of LIPO and AdaLIPO, introduced in the paper [TODO]().
A demo of these algorithms is available on the [IPOL website](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000391).

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