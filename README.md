# Project on Watermarking Deep Neural Networks # 


## Introduction
The code in this project serves as a practical implementation of the theoretical concepts discussed in the "Digital Watermarking" section of the work. Thereby, the author consulted existing approaches and compiled a repository with working code to illustrate the embedding and extraction process of digital watermarks in a Deep Neural Network (DNN). In the following, the contents of the repository are outlined, including a main black-box setting and two support programs (Rouhani, Chen, & Koushanfar, 2019; Quan, Teng, Chen, & Ji, 2020). These additional programs define the functionality of the main program and the architecture of the underlying DNN. Finally, a Wide Residual Network (WRN) topology is illustrated, aiming at a better understanding of the complex structure DNNs can possess (Uchida, Nagai, Sakazawa, & Satoh, 2017).

## Watermarking a DNN in a Black-Box Setting
The main procedure of this black-box setting involves importing and training a dataset (in this case MNIST) and embedding an owner-specific Watermark (WM) in specific layers of the DNN model that is used to train the data. This DNN was previously defined in DeepMarks.py and topology.py, whose specific functions are described in the following sections. In the end, the black-box watermarking framework detects the WM information and outputs a Boolean decision (Theta) on whether or not the marked model is correctly authenticated by owner (Rouhani et al., 2019). <br />
Thereby, a possible output scenario where the DNN has indeed been replicated by a malicious third-party user might look like this (NOTE: accuracy of the marked model is at 98.65%, indicating no degradation from the unmarked model):
```bash
60800/60800 [==============================] - 5s 86us/step - loss: 0.6138 - accuracy: 0.9865 - val_loss: 0.2926 - val_accuracy: 0.9837
Usable key len is:  74
WM key generation finished. Save watermarked model. 
Probability threshold p is  0.0001
Mismatch threshold is :  12
Mismatch count of marked model on WM key set =  0
If the marked model is correctly authenticated by owner:  True
```
The requirements to run this program in Python are: Keras 2.3.1, tensorflow 2.3.1, numpy, matplotlib and pandas:
```bash
pip install keras==2.3.1
pip install tensorflow==2.3.1
pip install tensorflow-gpu==1.14.0
```

## DeepMarks: The Functionality of the Black-Box
DeepMarks' main purpose is defining the functions which are called by the black-box setting. These functions include the computation of the mismatch threshold, the key generation and, finally, a counter for the response mismatch, allowing the main program the determination of the model's original owner.

## DNN Topology
This program defines the model internal of the DNN used in this black-box setting. Thereby, a ReLU activation function is applied throughout most layers of the DNN, while a softmax function is utilized as the last activation function to normalize the output to a probability distribution over predicted output classes.

## WRN-28-8
The Neural Network shown below is a WRN-28-8 model. This model and its more powerful brother, the WRN-28-10 model, possess ideal architectures for image recognition on large datasets, e.g. CIFAR10. The network structure of a WRN is thereby powerful enough to outperform even the deepest residual networks, since the issue of diminishing feature reuse does not occur (Majumdar, Denouden, Uchida, Saha, & Moser, 2018).

<p align="center"> 
<img src="https://github.com/DeepMarks/DNN-Watermarking/blob/main/images/WRN-28-8.png">
</p>
<br />

## References
[1] Majumdar, S., Denouden, T., Uchida, Y., Saha, D., & Moser, N. (2018, June 25). Wide Residual Networks in Keras. Retrieved October 14, 2020, from https://github.com/titu1994/Wide-Residual-Networks <br />
[2] Quan, Y., Teng, H., Chen, Y., & Ji, H. (2020). Watermarking Deep Neural Networks in Image Processing. IEEE Transactions on Neural Networks and Learning Systems, 1-14. doi:10.1109/TNNLS.2020.2991378 <br />
[3] Rouhani, B. D., Chen, H., & Koushanfar, F. (2019, April 14). Deepsigns: An End-to End Watermarking Framework for Ownership Protection of Deep Neural Networks. Retrieved August 19, 2020, from https://github.com/Bitadr/DeepSigns <br />
[4] Uchida, Y., Nagai, Y., Sakazawa, S., & Satoh, S. (2017, July 30). Embedding Watermarks into Deep Neural Networks. Retrieved July 14, 2020, from https://github.com/yu4u/dnn-watermark
