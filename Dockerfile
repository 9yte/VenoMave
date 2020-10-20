FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update \
    && apt upgrade -y \
    && apt install -y tmux htop wget gdb
    
RUN apt update && apt install -y sox libsox-dev libsox-fmt-all

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update \
    &&  apt install -y python-pip python3-pip \
    &&  pip3 install -r /tmp/requirements.txt \
    &&  rm /tmp/requirements.txt

RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchaudio

# COPY montreal-forced-aligner /root/asr-python/montreal-forced-aligner
# COPY digits.dict /root/asr-python/digits.dict

# fix montreal forced aligner issue
RUN apt install -y libgfortran3:amd64 

RUN pip3 install higher
RUN pip3 install ipython

RUN apt install -y git
