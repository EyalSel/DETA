Install libraries:
conda create -n deformable_detr python=3.7
conda install cudatoolkit==11.3.1 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ./models/ops
# prepend CUDA_HOME=/usr/local/cuda... with an 11.X version of CUDA if it's not set by default in nvidia-smi
sh ./make.sh
# unit test (should see all checking is True)
python test.py



Download weights:
https://utexas.box.com/s/py0nnzl5fmx4bw3hx7p2ww038uyznjs0

Save as "adet_swin_fth.pth"

