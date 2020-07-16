FROM ubuntu

# System requirements
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  python3-pip \
  swig \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pytest==4.6.* pep8 codecov pytest-cov flake8 flaky
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt \
  | xargs -n 1 -L 1 pip3 install

# Install
RUN pip3 install \
  auto-sklearn \
  jupyter
