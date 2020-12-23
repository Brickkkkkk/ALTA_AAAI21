# Teaching Active Human learners
The code of our AAAI21 paper "Teaching Active Human Learners" is mainly modified on the [code](https://github.com/macaodha/explain_teach) of a [CVPR'18 paper](https://arxiv.org/abs/1802.06924).

The main modifications are as follows:
- Add an ALTA teacher model in ./code/teach/offline_teachers.py
- Add a review mechanism in the teaching interface in ./code/teachingapp/templates/teaching.html

# Teaching Categories to Human Learners with Visual Explanations
Code for recreating the results in our CVPR 2018 paper.

`code` contains the main code for the teaching algorithms and data generation.  
`data` contains the image datasets.  
`results` contains the results files and plot generation scripts.   


## Reference
If you find our work useful in your research please consider citing our paper.  
```
@inproceedings{explainteachcvpr18,
  title     = {Teaching Categories to Human Learners with Visual Explanations},
  author    = {Mac Aodha, Oisin and Su, Shihan and Chen, Yuxin and Perona, Pietro and Yue, Yisong},
  booktitle = {CVPR},
  year = {2018}
}
```
