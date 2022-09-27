# We choose ubuntu 22.04 as our base docker image

FROM ubuntu:22.04

# We install pip and git from https://packages.ubuntu.com/jammy/apt
RUN apt-get update && \
    apt-get install -y python3-pip git

# We upgrade pip and install setuptools
RUN pip3 install pip setuptools --upgrade

# We remove the version of setuptools install via apt, as it is outdated
RUN apt-get purge -y python3-setuptools


# We set the working directory to install docker dependencies
WORKDIR /tmp/

# We clone the package repo from github and install it and the binder dependencies
RUN git clone https://github.com/jorgensd/reproducibility && \
    pip install ./reproducibility[binder]

# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp

# Create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy home directory for usage with Binder
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
ENTRYPOINT []