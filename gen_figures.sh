#/bin/sh

set -x

python main.py --function himmelblau.py -n 2000 -r 10 && \
python main.py --function holder.py -n 2000 -r 10 && \
python main.py --function rastrigin.py -n 2000 -r 10 && \
python main.py --function rosenbrock.py -n 2000 -r 10 && \
python main.py --function sphere.py -n 40 -r 10 && \
python main.py --function square.py -n 500 -r 10