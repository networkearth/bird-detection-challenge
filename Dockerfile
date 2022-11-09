FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential

# add python and libraries
RUN apt-get -y install python3.9 && \
	apt-get -y install python3-pip && \
	pip3 install --upgrade setuptools pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN apt-get -y install python3.9-dev

RUN apt-get -y install libsndfile1-dev

RUN pip3 install tensorflow==2.10.0
RUN pip3 install matplotlib==3.6.1 \
                    librosa==0.9.2 \
                    pandas==1.5.1 \
                    click==8.1.3 \
                    tqdm==4.64.1

# add git, vim and curl
RUN apt-get -y install git

