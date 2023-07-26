#!/bin/bash

# Use CACHE_DIR to cache downloaded wheels from PyPI.
if [ -z "${CACHE_DIR+x}" ]; then
    DOCKER_ARGS=
else
    DOCKER_ARGS="-v $CACHE_DIR/pip:/root/.cache/pip"
fi

# Use custom manylinux2014 image with CUDA support.
DOCKER_IMAGE=doge.skoltech.ru/manylinux2014_x86_64

# Build wheels across version matrix.
CUDA_VERSIONS=(10.2 11.1 11.3 11.5)
PYTHON_VERSIONS=(3.8 3.9 3.10)
TORCH_VERSIONS=(1.10.2 1.11.0)
for CUDA_VERSION in ${CUDA_VERSIONS[@]}; do
    for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
        for TORCH_VERSION in ${TORCH_VERSIONS[@]}; do
            versions="$CUDA_VERSION $PYTHON_VERSION $TORCH_VERSION"

            docker run --rm -ti \
                $DOCKER_ARGS \
                -v $(pwd):/workspace \
                -w /workspace \
                $DOCKER_IMAGE:$CUDA_VERSION \
                /workspace/ci/build-wheel.sh $versions

            if [ $? -ne 0 ]; then
                echo
                echo "ERROR Failed to build wheels for versions $versions."
                exit 1
            fi
        done
    done
done

echo
echo "OK All wheels are built!"
