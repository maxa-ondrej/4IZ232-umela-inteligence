FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y git texlive librsvg2-bin
RUN apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-extra-utils texlive-latex-extra texlive-xetex texlive-lang-greek
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app
COPY ./script.sh ./app/
RUN chmod +x /app/script.sh

ENV WORKING_DIR=$PWD

ENTRYPOINT [ "/app/script.sh" ]
