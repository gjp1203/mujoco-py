# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
#FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1

FROM docker.io/tensorflow/tensorflow:latest-gpu

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev xvfb \
    ffmpeg curl \ 
    patchelf \
    libglfw3 \
    libglfw3-dev \
    cmake \
    zlib1g \
    zlib1g-dev \
    swig \
    tmux \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
#RUN virtualenv --python=python3.6 env

#RUN rm /usr/bin/python
#RUN ln -s /env/bin/python3.6 /usr/bin/python
#RUN ln -s /env/bin/pip3.6 /usr/bin/pip
#RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux  /root/.mujoco/mujoco200 \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/
COPY ./requirements.dev.txt /mujoco_py/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY . /mujoco_py
RUN python setup.py install
