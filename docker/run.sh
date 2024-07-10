#!/bin/bash

# Build the dockerfile 
if [ -z "$(docker images -q sad_vio)" ]; then
  docker build -f Dockerfile . -t sad_vio;
fi

# Allow X server connection
xhost +local:root
docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    sad_vio
# Disallow X server connection
xhost -local:root
