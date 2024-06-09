FROM python:3.10-slim-bullseye

## Pip dependencies
# Upgrade pip
RUN pip install --upgrade pip

# Install production dependencies
RUN pip install tensorflow==2.15.0

# Add non-root user
ARG USERNAME=nonroot

RUN groupadd --gid 1000 $USERNAME; \ 
    useradd --uid 1000 --gid 1000 -m $USERNAME; \
    true
    
## Make sure to reflect new user in PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
USER $USERNAME