# Define build arguments
ARG USERNAME=nonroot

FROM mcr.microsoft.com/vscode/devcontainers/python:3.10-bullseye

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

