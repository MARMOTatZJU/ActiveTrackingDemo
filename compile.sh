#!/usr/bin/env bash
echo "Running in Environment with CUDA."
pushd third_party
  # compile in video_analyst
  pushd video_analyst
  bash compile.sh
  popd
  # CenterNet
  pushd CenterNet
    # compile DCNv2
    pushd src/lib/models/networks/DCNv2
    sudo ./make.sh
    popd
    # compile NMS
    pushd src/lib/external
    make
    popd
  popd
popd
