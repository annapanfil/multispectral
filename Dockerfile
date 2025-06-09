FROM ultralytics/ultralytics:latest-jetson-jetpack4

RUN apt-get update && apt-get install -y \
    libimage-exiftool-perl \
    iputils-ping \
    vim

RUN pip install setuptools==65.5.0

RUN pip install --no-cache-dir \
    click \
    pyexiftool \
    omegaconf \
    scikit-image \
    pysolar \
    pyzbar

RUN echo 'export PATH=/home/lariat/code/multispectral/src:$PATH' >> /root/.bashrc && \
    echo 'export PYTHONPATH=/home/lariat/code/multispectral/libraries/imageprocessing:$PYTHONPATH' >> /root/.bashrc

RUN mkdir -p /home/lariat
WORKDIR /home/lariat
RUN wget https://github.com/exiftool/exiftool/archive/refs/tags/12.40.tar.gz && \
    tar -xzf 12.40.tar.gz && \
    ln -s /home/lariat/exiftool-12.40/exiftool /usr/local/bin/exiftool

CMD ["bash", "--rcfile", "/root/.bashrc"]
