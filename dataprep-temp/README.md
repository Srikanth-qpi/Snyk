# QpiAI Pro - Data Preparation
QpiAI Pro Data Preparation Microservice

## Technical Overview


## Setup Instructions

1. You can either do all other setup by yourself or you can use the script by running. It will automatically initialize git repo and create the main and dev branch and setup the pre-commit git hooks assuming you installed pre-commit already

```bash
make init
```

> If you find yourself not having the make command line utility, you can install it by `sudo apt install make` in ubuntu.

2. Setting up poetry private repository additions. Since we are using *qpiai_utils*, our private repository, we are required access credentials to download our python package.

```bash
poetry config repositories.foo https://gitlab.qpiai.tech/api/v4/projects/4/packages/pypi/simple
poetry config http-basic.foo qpiai-pro-backend cP_Af946R2B2Ut9yRg_g
```

3. Install dependency and intialize poetry virtualenv

```bash
poetry install
```

> You can see .venv folder create in the root of the project

## Development Instructions

1. You can source the poetry's project virtualenv by
```bash
poetry shell
```

> You should do this setup everytime you open a new terminal shell session.

2. To start the logging server databases (MongoDB, Redis), use docker-compose to start the database servers in a docker containers

```bash
docker-compose up -d
```

> Note: You can only see your outputs stored in the mongodb database. So it is necessary to start your database server before FastAPI Server. Also you can install [robo3T](https://robomongo.org), a MongoDB GUI exploration tool to see the responses.

2. Now you can either execute the below commands in order to start the FastAPI Server or you can use make tool

```bash
export PYTHONPATH=$(pwd)
python app/main.py
```

or 

```bash
make start
```

## Deployment Variables Configuration

```bash
ENV_MODE = production # It clears out some cache created while building the app during containerization
REDIS_HOST = localhost # Redis Server IPv4 Address
REDIS_PORT = 6379 # Redis Server Port

MONGODB_USER = qpiai #MongoDB Server Login Username
MONGODB_PASS = qpiai # MongoDB Server Login Password
MONGODB_HOST = localhost # MongoDB Server IPv4 Address
MONGODB_PORT = 27017 # MongoDB Server Port
MONGODB_NAME = admin # MongoDB Server Login Database
```

## Goals

