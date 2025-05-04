# Use NVIDIA's PyTorch image for CUDA 11.1 + cuDNN 8, which bundles
# Python 3.8 and torch==1.9.0+cu111, torchvision==0.10.0+cu111
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Create and set the working directory
RUN mkdir -p /DepthPrompting
WORKDIR /DepthPrompting

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
      h5py \
      tqdm \
      scipy \
      matplotlib \
      nuscenes-devkit \
      imageio \
      pillow==9.5.0 \
      pandas \
      opencv-python && \
    pip install --no-cache-dir \
      mmcv-full==1.3.13 \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Copy your code into the container

COPY . . 

# Arguments for user creation (can be passed at build time)
ARG USERNAME=admin
ARG USER_ID=1000
ARG GROUP_ID=1000

# Install sudo and other necessary packages
RUN apt-get update && apt-get install -y sudo

# Create a user with the specified UID and GID
RUN groupadd -g $GROUP_ID $USERNAME && \
    useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USERNAME

# Add the user to the sudoers file with full root access without a password
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# RUN usermod -aG sudo,video,i2c "$USERNAME"
RUN usermod -aG sudo,video "$USERNAME"

# Set the default user to the new user
USER $USERNAME

# Default entrypoint
CMD ["bash"]
