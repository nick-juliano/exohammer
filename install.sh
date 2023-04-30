conda create -n $1
conda activate exo_env
conda install pip
conda install python=3.8
pip install .