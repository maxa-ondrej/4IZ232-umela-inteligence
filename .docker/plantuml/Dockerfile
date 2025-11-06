FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y git wget openjdk-17-jre graphviz librsvg2-bin
RUN rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/plantuml/bin
RUN wget -O /opt/plantuml/bin/plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2025.4/plantuml-1.2025.4.jar

RUN mkdir -p /app
WORKDIR /app
ENV WORKING_DIR=/app

COPY ./script.sh ./app/
RUN chmod +x /app/script.sh

ENTRYPOINT [ "/app/script.sh" ]
