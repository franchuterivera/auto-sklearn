FROM ubuntu:18.04

WORKDIR /auto-sklearn

# Copy the checkout autosklearn version for installation
ADD . /auto-sklearn/


# install linux packages
RUN apt-get update
# https://github.com/automl/auto-sklearn/issues/314
RUN apt-get remove swig
RUN apt-get install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig

RUN apt install -y python-dev python-pip
RUN pip3 install --upgrade setuptools

# System requirements
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pytest==4.6.* pep8 codecov pytest-cov flake8 flaky openml
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
#RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | LC_ALL=C.UTF-8 xargs -n 1 -L 1 pip3 install
RUN pip3 install jupyter

# Install
RUN ls /
RUN ls /auto-sklearn/
RUN pip3 install -e /auto-sklearn/
