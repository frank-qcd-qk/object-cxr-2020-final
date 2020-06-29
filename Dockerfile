#
# Darknet GPU Configuration File
# @author Frank Chude Qian (frankq at ieee dot com)
# v1.0.0
# 
# Copyright (c) 2020 Chude Qian - https://github.com/frank-qcd-qk
#

#! The base image from nvidia with cuda and cudnn
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="frank1@ieee.org"

#! Set working directory
WORKDIR /workENV

#! Software install
RUN \
    apt-get update && apt-get install -y \
    gcc \
    git \
    git-lfs \
    vim \
    build-essential \
    python3 \
    python3-pip \
    libsm6 \ 
    libxext6 \
    libxrender-dev \ 
    wget && \
    rm -rf /var/lib/apt/lists/*

#! Get Darknet
RUN \
    git clone https://github.com/AlexeyAB/darknet

WORKDIR darknet/

#! Modify Make File
RUN sed -i s/GPU=0/GPU=1/g Makefile
RUN sed -i s/CUDNN=0/CUDNN=1/g Makefile
RUN sed -i s/LIBSO=0/LIBSO=1/g Makefile
RUN make 

#! Copy core library
WORKDIR /workENV
COPY src/ ./src/

RUN rm -rf src/darknetCore/libdarknet.so
RUN mv darknet/libdarknet.so src/darknetCore/libdarknet.so
RUN rm -rf darknet/

#! PIP time!

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
RUN rm requirements.txt


#！ Test Nvidia
CMD ["/workENV "]