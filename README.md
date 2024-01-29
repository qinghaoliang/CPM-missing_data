# Connectome-based Predictive Modeling with Missing Data 
Our manuscript is available on bioRxiv https://www.biorxiv.org/content/10.1101/2023.06.09.544392v1.full.pdf and published at Imaging Neuroscience https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00071/118937/Rescuing-missing-data-in-connectome-based?utm_term=&utm_campaign=&utm_source=adwords&utm_medium=ppc&hsa_acc=4298009839&hsa_cam=20485121991&hsa_grp=153323141712&hsa_ad=671196874738&hsa_src=g&hsa_tgt=dsa-2189678350017&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=CjwKCAiAtt2tBhBDEiwALZuhAADnfzje3gv8v7VwWvOXvQWDzSkWgQ32SboQBfZT0XbuT9yTEzDBiBoCKRsQAvD_BwE

## Dependencies
Missing Connectome (Python)
- numpy
- scikit-learn

Missing Phenotyes (R)
- glmnet
- cvTools
- missMDA
- missForest
- mice
- VIM
- plyr

## Data
The dataset should include both connectome data and phenotypical data. The data should be organized in the same structure as demonstrated in the following scripts of the experiments.

## Experiment 
You could set up your experiment like the example code below and run:
### Experiment 1: CPM with missing connectomes (python)
```
python Expe_coms_behav.py -sd 10
```

### Experiment 2: CPM with missing connectomes and phenotypes (R)
```
Rscript Experiment_com_behav.R 10
```

## Reference
Qinghao Liang, Rongtao Jiang, Brendan D. Adkinson, Matthew Rosenblatt, Saloni Mehta, Maya L. Foster, Siyuan Dong, Chenyu You, Sahand Negahban, Harrison H. Zhou, Joseph Chang, Dustin Scheinost; Rescuing missing data in connectome-based predictive modeling. Imaging Neuroscience 2024; doi: https://doi.org/10.1162/imag_a_00071


Please cite our paper if you find this repository useful:
```
@article{10.1162/imag_a_00071,
    author = {Liang, Qinghao and Jiang, Rongtao and Adkinson, Brendan D. and Rosenblatt, Matthew and Mehta, Saloni and Foster, Maya L. and Dong, Siyuan and You, Chenyu and Negahban, Sahand and Zhou, Harrison H. and Chang, Joseph and Scheinost, Dustin},
    title = "{Rescuing missing data in connectome-based predictive modeling}",
    journal = {Imaging Neuroscience},
    year = {2024}
}
```