<h1 style="margin: 0 0 0.35 rem 0; line-height: 1.1;">MixFlow</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#)
[![Project status](https://img.shields.io/badge/Status-Research%20code-informational.svg)](#)
[![Docs](https://img.shields.io/badge/Docs-local-blueviolet.svg)](./docs/index.md)

Mixture-Conditioned Flow Matching for Out-of-Distribution Generalization.

---
<table>
<tr>
<td align="center">
<img src="./docs/figs/manif_vanila_B.png" width="100%">
<br>
<b>(A)</b> Vanilla CFM
</td>
<td align="center">
<img src="./docs/figs/manif_ptflow_B.png" width="100%">
<br>
<b>(B)</b> MixFlow
</td>
</tr>
</table>

## Overview

MixFlow is a conditional flow-matching framework for descriptor-controlled generation. Instead of relying on a single Gaussian base distribution, MixFlow learns a mixture base and a descriptor-conditioned flow jointly, trained via shortest-path flow matching. This joint modeling is designed to extrapolate smoothly to unseen conditions and improve out-of-distribution generalization across tasks.

## Publication

This project is based on the **MixFlow** manuscript.

- **Title:** MixFlow: Mixture-Conditioned Flow Matching for Out-of-Distribution Generalization
- **Authors:** Andrea Rubbi, Amir Akbarnejad, Mohammad Vali Sanian, Aryan Yazdan Parast, Hesam Asadollahzadeh, Arian Amani, Naveed Akhtar, Sarah Cooper, Andrew Bassett, Lassi Paavolainen, Pietro Liò, Sattar Vakili, Mo Lotfollahi
- **Link:** _TODO_

## Datasets

### Synthetic Data

We construct a synthetic benchmark of letter populations, where each condition corresponds to a letter and a specific rotation. Each descriptor encodes the letter identity and rotation, and MixFlow learns a mixture base distribution per condition. This setup allows us to test extrapolation to unseen letters and rotation angles.

### Morphological Perturbations

We evaluate MixFlow on high-content imaging data in feature space. Cells (from BBBC021 and RxRx1) are embedded with a vision backbone, and the model is trained to generate unseen phenotypic responses from compound descriptors alone.

### Perturbation Datasets

For transcriptomic perturbations, we use Chemical- or CRISPR-based single-cell datasets (Norman, Combosciplex, Replogle and iAstrocytes). Conditions correspond to perturbations' embeddings from pretrained models, and MixFlow is trained to model the distribution of perturbed cells.

## Documentation

Check the <a href="./docs/index.md"> documentation </a> for more information about how to use the model and get the data.

## License 

This work is released with the MIT license, please see <a href="./LICENSE"> the license file </a> for more information.

## Authors

Andrea Rubbi, Amir Akbarnejad, Mohammad Vali Sanian, Aryan Yazdan Parast, Hesam Asadollahzadeh, Arian Amani, Naveed Akhtar, Sarah Cooper,
Andrew Bassett, Pietro Liò, Lassi Paavolainen, Sattar Vakili,
Mo Lotfollahi

