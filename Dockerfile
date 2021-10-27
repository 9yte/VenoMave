FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt update \
    && apt upgrade -y \
    && apt install -y tmux htop wget gdb unzip
    
# Install Matlab runtime (@ /usr/local/MATLAB/MATLAB_Runtime/v96)
# (download @ https://de.mathworks.com/products/compiler/matlab-runtime.html)

RUN mkdir /matlab && \
    wget https://ssd.mathworks.com/supportfiles/downloads/R2019a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019a_Update_6_glnxa64.zip -O /matlab/matlab_runtime.zip && \
    apt update && apt install -y libxrender1 libxt6 libxcomposite1 && \
    unzip /matlab/matlab_runtime.zip -d /matlab/matlab_runtime && \
    /matlab/matlab_runtime/install -mode silent -agreeToLicense yes && \
    rm -rf /matlab/matlab_runtime*
ADD hearing_thresholds/_compiled /matlab/hearing_thresholds

RUN apt update && apt install -y sox libsox-dev libsox-fmt-all

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update \
    &&  apt install -y python-pip python3-pip \
    &&  pip3 install -r /tmp/requirements.txt \
    &&  rm /tmp/requirements.txt

# RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install torchaudio==0.6.0
RUN pip3 install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# COPY montreal-forced-aligner /root/asr-python/montreal-forced-aligner
# COPY digits.dict /root/asr-python/digits.dict

# fix montreal forced aligner issue
RUN apt install -y libgfortran3:amd64

RUN pip3 install higher
RUN pip3 install ipython

RUN apt install -y git