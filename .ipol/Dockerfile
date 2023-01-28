FROM python:3.9

RUN pip install --no-cache-dir pip==22.0.4

COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/*

ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pwd && ls -l

#RUN opam init -y && eval $(opam env) && opam install -y dune menhir


