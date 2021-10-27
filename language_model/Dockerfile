FROM pykaldi/pykaldi:latest

WORKDIR /pykaldi/examples/setups/aspire/
RUN ./models.sh 

RUN cp -R /pykaldi/examples/setups/aspire /root/aspire \
    && rm -rf /root/aspire/data/lang /root/aspire/align.py /root/aspire/path.sh 

ADD pykaldi_patches/* /root/aspire/
WORKDIR /root/aspire/
