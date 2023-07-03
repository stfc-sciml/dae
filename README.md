# Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations

Jaehoon Cha, Jeyan Thiyagalingam

This is the official implementation of the paper "Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations".

Please create xyc datasets first using make_xyc_dataset.py file to run the 'main.py'. 


This repository is organised as follows:

```
├── dataset.py/              <XYC dataset loader> 
├── make_xyc_dataset.py           <To creat the XYC dataset>
├── main.py/                 <Implementing the code> 
├── model.py/               <DAE model architecture> 
├── plot.py/               <Plotting latent traversals and latent space> 
└── sl
```



Requirements are  

* ``torch``
* ``numpy``, and
* ``matplotlib``.
