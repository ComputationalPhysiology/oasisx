# We choose ubuntu 22.04 as our base docker image

FROM dolfinx/dolfinx:v0.6.0-r1


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# We set the working directory to install docker dependencies
WORKDIR /tmp/

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

RUN python3 -m pip install .[docs,test]

USER ${NB_USER}
ENTRYPOINT []