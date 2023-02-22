# 3D-FaceReconstruction


This code is developed on the top of [MICA](https://arxiv.org/pdf/2204.06607) for 3D Face reconstruction, [SwAV](https://arxiv.org/abs/2006.09882) for Self-supervised technique. 
we thank to their efficient and neat codebase. 


## Install requirements 

```bash
$ pip install requirements.txt
```



## Evaluate Self-supervised model's weights 
```
python -m torch.distributed.launch --nproc_per_node=1 mask_deepcluster_universal.py
```

## Evaluate Downstream tasks 
To train **SSL3D** model, run the following command: 
```
python -m torch.distributed.launch --nproc_per_node=1 mask_deepcluster_universal.py

python -m torch.distributed.launch --nproc_per_node=1 mask_deepcluster_usst.py
```

Example output during training :
```
INFO - 02/08/23 04:00:35 - 0:03:54 - Epoch: [0][350]    Time 0.612 (0.639)     Data 0.000 (0.006)       Loss 0.0001 (0.0293)    Lr: 0.2047
```

## Citation

If you find our work is useful in your research, please consider citing:

```bibtex
@misc{,
    title   = {}, 
    author  = {},
    year    = {2023},
    eprint  = {},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```


## Contact

If you have any questions or concerns, feel free to send mail to [hoangng210a@gmail.com](mailto:hoangng210a@gmail.com).
I will try our best to answer all of your concerns ASAP. 
