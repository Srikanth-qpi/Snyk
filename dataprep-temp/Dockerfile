FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 AS pro-base-stage

LABEL vendor="qpiai.tech"

# Installing Driver Utils
RUN apt-get update && apt install -y --no-install-recommends software-properties-common

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    libssl-dev\
    curl

RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y python3-virtualenv

#RUN apt-get update -qq && apt-get install -qqy \
    #apt-transport-https \
    #ca-certificates \
    #curl \
    #lxc \
    #iptables

RUN curl -sSL https://get.docker.com/ | sh

FROM pro-base-stage AS pro-builder-base

ARG ENV_MODE

ENV  \
  # python :
  PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PYTHONDONTWRITEBYTECODE=1 \
  # pip :
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # poetry:
  POETRY_VERSION=1.1.7 \
  POETRY_HOME="/opt/poetry"\
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  #Our Project Setup
  PROJECT_PATH="/opt/project-app" \
  VENV_PATH="/opt/project-app/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get -y install libsndfile1

FROM pro-builder-base as pro-development

ENV FASTAPI_ENV=development

ENV DATASET_MOUNT=/mnt

WORKDIR ${PROJECT_PATH}

WORKDIR /qpiai-app

COPY app/dist/main main

#RUN apt-get update && apt-get install -y python3-pyparsing 

WORKDIR /qpiai-app/main/

ENV PYTHONPATH=/qpiai-app

CMD ["./main"]
