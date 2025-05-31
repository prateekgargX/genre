# GenRe : Generative Models for Robust Algorithmic Recourse

<p align="center">
<!--   <a href="https://github.com/prateekgargX/genre/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a> -->
  <a href="https://openreview.net/forum?id=NtwFghsJne">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-blue">
  </a>
  <a href="https://arxiv.org/abs/2505.07351">
    <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2505.07351-b31b1b.svg">
  </a>
</p>

1. setup environment using `requirements.txt`. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. setup datasets

```bash
chmod +x ./scripts/setup_datasets.sh
./scripts/setup_datasets.sh
```

3. train true classifiers

```
python scripts/train_rf.py [options]
# example
python scripts/train_rf.py --dataset heloc
```

4. train ann classifiers

```
python scripts/train_ann.py [options]
# example
python scripts/train_ann.py --dataset heloc --lsrc rf --epochs 100 --batch 64 --lr 0.001 --device 0 --hidd 10 10 10
```

5. train pairmodel

```
python scripts/train_bpm.py [options]
```

6. Checkout `genre_sampler.ipynb` for running the code and evaluation.

To see available options, use `--help`

### Baselines

To run baselines, install CARLA from https://github.com/MartinPawel/ProbabilisticallyRobustRecourse/ and activate the environment[don't install GPU version of pytorch]

- Diverse Counterfactual Explanations (DiCE): [Paper](https://arxiv.org/pdf/1905.07697.pdf)
- Growing Sphere (GS): [Paper](https://arxiv.org/pdf/1910.09398.pdf)
- Wachter: [Paper](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
- ROAR: [Paper](https://proceedings.neurips.cc/paper/2021/hash/8ccfb1140664a5fa63177fb6e07352f0-Abstract.html)
- PROBE: [Paper](https://arxiv.org/pdf/2203.06768)
- CCHVAE: [Paper](https://arxiv.org/pdf/1910.09398)
- REVISE: [Paper](https://arxiv.org/pdf/1907.09615)
- CRUDS: [Paper](https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf)

### Datasets

- Adult: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- COMPAS: [Source](https://www.kaggle.com/danofer/compass)
- HELOC: [Source](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc/data)

### Citation

If you use this work, please cite:

```bibtex
@inproceedings{
garg2025sample,
title={From Search to Sampling: Generative Models for Robust Algorithmic Recourse},
author={Garg, Prateek and Nagalapatti, Lokesh and Sarawagi, Sunita},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=NtwFghsJne}
}
```

## Contact

Prateek Garg ([prateekg@iitb.ac.in](prateekg@iitb.ac.in))
