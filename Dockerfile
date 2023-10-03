FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_VERSION=12.0
ENV VULKAN_SDK_VERSION=1.3.236.0

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y cmake  && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y python3.10 && \
    apt-get install -y python3-pip && \
    apt-get install -y libx264-dev && \
    apt-get install -y python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# VULKAN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1  \
    libgles2  \
    libxcb1-dev \
    wget \
    vulkan-utils \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/latest/linux/vulkan_sdk.tar.gz \
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \ 
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan
ENV NVIDIA_DRIVER_CAPABILITIES all
    
# User
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" nmpc && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" nmpc
    
COPY --chown=nmpc:nmpc asound.conf /etc/

USER nmpc

# Formula-Student-Driverless-Simulator should be in run dir, along with t_renderer and asound.conf
COPY --chown=nmpc:nmpc . /home/nmpc/
WORKDIR /home/nmpc/
RUN pip3 install --upgrade pip
RUN pip3 --default-timeout=60 install -r requirements.txt
    
# Acados installation
RUN git clone https://github.com/acados/acados.git && \
 cd acados && \
 git submodule update --recursive --init && \
 mkdir -p build && \
 cd build && \
 cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON .. && \
 make install -j4
RUN pip3 install -e /home/nmpc/acados/interfaces/acados_template
RUN mv t_renderer /home/nmpc/acados/bin
ENV ACADOS_SOURCE_DIR /home/nmpc/acados
ENV LD_LIBRARY_PATH /home/nmpc/acados/lib
