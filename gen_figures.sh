#/bin/sh

set -x

python main.py --function="-((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)" -n 2000 -b -4 4 -4 4 -k 283 --name "himmelblau" && \
python main.py --function="np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - (np.sqrt(x[0]**2 + x[1]**2)) / np.pi)) )" -n 2000 -b -10 10 -10 10 -k 30 --name "holder" && \
python main.py --function="-(10 * 2 + (x[0]**2 - 10 * np.cos(2*np.pi*x[0])) + (x[1]**2 - 10 * np.cos(2*np.pi*x[1])))" -n 1000 -b -5.12 5.12 -5.12 5.12 -k 96 --name "rastrigin" && \
python main.py --function="9 * np.sin(20 * x**(6/7)) * np.sin(4 * x**(3/2))" -n 500 -b 0 10 -k 200 --name "rng" && \
python main.py --function="-(100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)" -n 5000 -b -3 3 -3 3 -k 14607 --name "rosenbrock" && \
python main.py --function="-np.sqrt( (x[0] - np.pi / 16)**2 + (x[1]-np.pi/16)**2 )" -n 30 -b 0 1 0 1 -k 1.5 --name "sphere" && \
python main.py --function="-(x**2)" -n 500 -b -10 10 -k 20 --name "square"