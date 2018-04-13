# Introduction

This repository implements [ACL 2017 Riemannian optimization for skip-gram negative sampling (Fonarev, Hrinchuk et al.)](https://arxiv.org/pdf/1704.08059.pdf).
```
@inproceedings{fonarev2017ro_sgns,
  title={Riemannian Optimization for Skip-Gram Negative Sampling},
  author={Alexander Fonarev and Oleksii Hrinchuk and Gleb Gusev and Pavel Serdyukov and Ivan Oseledets},
  booktitle={ACL},
  year={2017}
}
```

# Prerequisits

- [numpy](http://www.numpy.org)
- [scipy](https://www.scipy.org)
- [pandas](https://pandas.pydata.org)
- [gensim](https://radimrehurek.com/gensim/)
- [nltk](https://www.nltk.org)
- [bs4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

```
pip install numpy scipy pandas gensim nltk bs4 
```

# Usage

- Download [enwik9](http://mattmahoney.net/dc/enwik9.zip) dataset and preprocess raw data with Perl script [main_.pl](main_.pl). 
- Run IPython notebook [enwik_experiments.ipynb](enwik_experiments.ipynb).

# Algorithm


<div>
<figure>
<img src="/img/ro.png" width="40%" hspace="7%">
<img src="/img/algorithm.png" width="40%">
</figure>
</div>

<div class="container" style="width: 100%;">
 <div class="theme-table-image col-sm-6">
   <img src="/img/ro.png">
 </div>
 <div class="theme-table-image col-sm-6">
   <img src="/img/ro.png">
 </div>
</div>
