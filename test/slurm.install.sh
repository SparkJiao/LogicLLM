#!/bin/bash
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r

conda config --set show_channel_urls yes

conda create -n torch1.8 python=3.9

conda install pytorch==1.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge


pip install hydra-core
pip install
#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
