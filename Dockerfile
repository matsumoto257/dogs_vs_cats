# 実行環境というよりは開発環境
FROM python:3
USER root

WORKDIR /root/work/

RUN apt-get update && apt-get install -y vim less && pip install --upgrade pip && pip install --upgrade setuptools
COPY . ./dogs_vs_cats/
