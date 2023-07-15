# Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations

Jaehoon Cha, and Jeyan Thiyagalingam

This is the official implementation of the paper "Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations".

Prior to using the model (demonstrated in the main.py script), please make sure that the XYC datasets are in place. This can be created using the script make_xyc_dataset.py.


This repository is organised as follows:

```
├── dataset.py                <Dataset loader>
├── make_xyc_dataset.py       <Script for creating the XYC dataset>
├── main.py                   <Example implementation of the model>
├── model.py                  <DAE model architecture>
└── plot.py                   <Script for plotting latent traversals and latent space>
```



Requirements are

* ``torch``
* ``numpy``, and
* ``matplotlib``.
