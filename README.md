<p align="center"> 
<img src="https://github.com/TheTrueMrbequiet/Boolean-Coin/blob/master/HSG%20Logo.jpg">
</p>
<br />

# Bachelor's Thesis: Project on Watermarking DNNs # 

<br />

<p align="center">
Noah Mamié (17-607-714)
</p>

<p align="center">
<b> Watermarking Deep Neural Networks &mdash; <br />
An Effective Strategy to Protect Intellectual Property Rights on Self-Learning Systems </b> <br />
November 16, 2020
</p>

<p align="center">
Prof. Dr. Naomi Haefner <br />
<i> Language: Python </i>
</p>
<br />


## Introduction
The code in this project serves as a practical implementation of the theoretical concepts discussed in the "Digital Watermarking" section of the thesis. Thereby, the author consulted existing approaches and compiled a repository with working code to illustrate the embedding and extraction process of digital watermarks into a Deep Neural Network (DNN). In the following, the contents of the repository are outlined, including a main black-box setting and two support programs. These additional programs define the functionality of the main program and the architecture of the underlying DNN. Finally, a Wide Residual Network (WRN) topology is illustrated, aiming at a better understanding of the complex structure DNNs can possess.

## Watermarking a DNN in a Black-Box Setting
The main procedure of this black-box setting involves importing and training a dataset (in this case MNIST-MLP) and embedding an owner-specific Watermark (WM) in specific layers of the model that is used to train the data. This DNN was previously defined in DeepMarks, whose specific functions are described in the next section. In the end, the black-box watermarking framework detects the WM information and outputs a Boolean decision on whether the marked model is correctly authenticated by owner. <br />
The requirements to run this program in Python are: Keras 1.1.2 (<1.2.0), tensorflow 0.12.1 (<1.0.0), numpy, matplotlib and pandas:
```bash
pip install keras==1.1.2
pip install tensorflow==0.12.1
pip install tensorflow-gpu==0.12.1
```

## DeepMarks: The Functionality of the Black-Box

## DNN Topology

## WRN-28-8
The Neural Network shown below is a WRN-28-8 model. This model and its more powerful brother, the WRN-28-10 model, possess ideal architectures to for image recognition on large datasets, e.g. CIFAR10. The network structure of a WRN is thereby powerful enough to outperform even the deepest residual networks, since the issue of diminishing feature reuse does not occur.

<p align="center"> 
<img src="https://github.com/DeepMarks/DNN-Watermarking/blob/main/images/WRN-28-8.png">
</p>
<br />

## References
[1] Majumdar, S., Denouden, T., Uchida, Y., Saha, D., & Moser, N. (2018, June 25). Wide Residual Networks in Keras. Retrieved October 14, 2020, from https://github.com/titu1994/Wide-Residual-Networks. <br />
[2] Rouhani, B. D., Chen, H., & Koushanfar, F. (2019, April 14). Deepsigns: An End-to End Watermarking Framework for Ownership Protection of Deep Neural Networks. Retrieved August 19, 2020, from https://github.com/Bitadr/DeepSigns. <br />
[3] Uchida, Y., Nagai, Y., Sakazawa, S., & Satoh, S. (2017, July 30). Embedding Watermarks into Deep Neural Networks. Retrieved July 14, 2020, from https://github.com/yu4u/dnn-watermark.
