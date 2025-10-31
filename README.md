



Builds on the Flamby benchmark. FLamby is a benchmark for cross-silo Federated Learning with natural partitioning,
currently focused in healthcare applications. The main changes / updates made were to the fed_benchmark.py file in FLamby/benchmarks and to the implementation of FL algorithms in the FLamby/ strategies directory to tailor them to the eICU dataset.

The FLamby package contains:

- Data loaders that automatically handle data preprocessing and partitions of distributed datasets.
- Evaluation functions to evaluate trained models on the different tracks as defined in the companion paper.
- Benchmark code using the utilities below to obtain the performances of baselines using different strategies.
:arrow_right:[The API doc is available here](https://owkin.github.io/FLamby):arrow_left:


## Installation

Usews anaconda and pip. You can install anaconda by downloading and executing appropriate installers from the [Anaconda website](https://www.anaconda.com/products/distribution), pip often comes included with python otherwise check [the following instructions](https://pip.pypa.io/en/stable/installation/). We support all Python version starting from **3.7**.

You may need `make` for simplification. The following command will install all packages used by all datasets within FLamby. If you already know you will only need a fraction of the datasets inside the suite you can do a partial installation and update it along the way using the options described below.
Create and launch the environment using:

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make install
conda activate flamby
```

To limit the number of installed packages you can use the `enable` argument to specify which dataset(s)
you want to build required dependencies for and if you need to execute the tests (tests) and build the documentation (docs):

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make enable=option_name install
conda activate flamby
```

where `option_name` can be one of the following:
cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests

if you want to use more than one option you can do it using comma
(**WARNING:** there should be no space after `,`), eg:

```bash
git clone https://github.com/owkin/FLamby.git
cd FLamby
make enable=cam16,kits19,tests install
conda activate flamby
```
Be careful, each command tries to create a conda environment named flamby therefore make install will fail if executed
numerous times as the flamby environment will already exist. Use `make update` as explained in the next section if you decide to
use more datasets than intended originally.

### Update environment
Use the following command if new dependencies have been added, and you want to update the environment for additional datasets:
```bash
make update
```

or you can use `enable` option:
```bash
make enable=cam16 update
```



## Quickstart

Follow the [quickstart section](./Quickstart.md) to learn how to get started with FLamby.

## Reproduce benchmark and figures from the companion article

### Benchmarks
The results are stored in flamby/results in corresponding subfolders `results_benchmark_fed_dataset` for each dataset.
These results can be plotted using:
```
python plot_results.py
```
which produces the plot at the end of the main article.

In order to re-run each of the benchmark on your machine, first download the dataset you are interested in and then run the following command replacing `config_dataset.json` by one of the listed config files (`config_camelyon16.json`, `config_heart_disease.json`, `config_isic2019.json`, `config_ixi.json`, `config_kits19.json`, `config_lidc_idri.json`, `config_tcga_brca.json`):
```
cd flamby/benchmarks
python fed_benchmark.py --seed 42 -cfp ../config_dataset.json
python fed_benchmark.py --seed 43 -cfp ../config_dataset.json
python fed_benchmark.py --seed 44 -cfp ../config_dataset.json
python fed_benchmark.py --seed 45 -cfp ../config_dataset.json
python fed_benchmark.py --seed 46 -cfp ../config_dataset.json
```
We have observed that results vary from machine to machine and are sensitive to GPU randomness. However you should be able to reproduce the results up to some variance and results on the same machine should be perfecty reproducible. Please open an issue if it is not the case.
The script `extract_config.py` allows to go from a results file to a `config.py`.
See the [quickstart section](./Quickstart.md) to change parameters.








## Acknowledgements
- [Owkin](https://www.owkin.com)
- [FLamby](https://owkin.github.io/FLamby)





