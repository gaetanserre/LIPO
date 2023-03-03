#/bin/sh

set -x

python src/main.py --function himmelblau.py -n 2000 -r 1 && \
python src/main.py --function holder.py -n 2000 -r 1 && \
python src/main.py --function rastrigin.py -n 1500 -r 1 && \
python src/main.py --function rosenbrock.py -n 2000 -r 1 && \
python src/main.py --function sphere.py -n 35 -r 1 && \
python src/main.py --function square.py -n 2000 -r 1
