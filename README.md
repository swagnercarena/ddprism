# DDPrism: A Data-Driven Prism

**DDPrism** is a JAX-based framework for solving multi-view source separation problems. It iteratively learns diffusion model priors which enable direct sampling from the joint posterior without making any explicit source assumptions. More details on the problem, the method, and the results can be found in [this paper](https://arxiv.org/abs/XXXX.YYYYY).

## Installation

DDPrism requires some additional dependencies to run the experiments:

### Core Installation (Minimal Dependencies)

For basic usage of DDPrism's core functionality (diffusion models, sampling, and utilities):

```bash
git clone https://github.com/swagnercarena/ddprism.git
pip install .
```

### Full Installation (All Dependencies)

For running experiments, training models, and using all features:

```bash
git clone https://github.com/swagnercarena/ddprism.git
pip install .[experiments]
```

or equivalently:

```bash
git clone https://github.com/swagnercarena/ddprism.git
pip install .[all]
```

### Development Installation

For development work:

```bash
git clone https://github.com/swagnercarena/ddprism.git
pip install -e .[experiments]
```
