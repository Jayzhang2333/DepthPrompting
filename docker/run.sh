#! /bin/bash

docker run --rm -it \
	--net=host \
    --privileged \
	--runtime nvidia \
	--user $(id -u) \
	--env=NVIDIA_DRIVER_CAPABILITIES=all \
	--env=DISPLAY \
	--env=QT_X11_NO_MITSHM=1 \
	--pid=host \
	--cap-add=SYS_ADMIN \
	--cap-add=SYS_PTRACE \
	--shm-size=16g \
	-e XDG_RUNTIME_DIR=/run/user/1000 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v ~/.docker_bash_history:/home/admin/.bash_history:rw \
	-v ~/.tmux.conf:/home/admin/.tmux.conf \
	-v $PWD/../:/DepthPrompting:rw \
	-v /media/jay/Lexar:/media/jay/Lexar:rw \
	-v ~/.docker_bash_history:/root/.bash_history \
  -v /etc/machine-id:/etc/machine-id:ro \
	-w /DepthPrompting \
  --name DepthPrompting \
  jay:DepthPrompting \
  bash
