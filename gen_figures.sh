#/bin/sh

set -x

python main.py --function himmelblau.py -n 2000 -r 2 && \
python main.py --function functions/holder.py -n 2000 -r 2 && \
python main.py --function functions/rastrigin.py -n 1000 -r 2 && \
python main.py --function functions/rng.py -n 500 -r 2 && \
python main.py --function functions/rosenbrock.py -n 5000 -r 2 && \
python main.py --function functions/sphere.py -n 30 -r 2 && \
python main.py --function square.py -n 500 -r 2