# CarNet - Deep learning for car classification -  Python, Pytorch, Stanford Cars Dataset
Developed a deep learning model that can classify images using ML Decoders and EfficientNet, a scalable family of models that can adapt to different resource constraints. I showed the model’s scalability by training it on a powerful server with 37 million parameters and achieving 90.9% accuracy.

## Abstarct
Object detection and classification applications have grown to become one of the most widely used application of machine learning, neural networks, and deep learning. In this project, the aim is to develop a model that can identify the brand and model of cars, which can be used in multiple fields of object detection. Learning and prediction are performed on the Stanford Cars Dataset (SCD), which contains numerous images of cars with their respective labels of the brand, the model name, and the year of release. The project aims to create a new model by meshing ML-Decoders for classification with a family of models as the backbone network given by EfficientNet, which addresses the challenges of scaling up the model to any given target resource constraints.

---
## Features
* **Simplified representation of CarNet**

  -  ![alt text](https://github.com/roniepatil/CarNet-Deep-learning-for-car-classification/blob/main/image_resources/CarNet_representation.png)
* **CarNet Layers Architecture using EfficientNet-B5 backbone with MLDecoder Classifier**

  - ![alt text](https://github.com/roniepatil/CarNet-Deep-learning-for-car-classification/blob/main/image_resources/CarNet_layers.png)


## Results
Fused two different deep learning architectures taken from the classification block of TResNet-L + MLDecoder implementation that achieves 96.41% accuracy on ImageNet, and the backbone of EfficientNet architectures that require training of significantly lesser number of parameters to predict the brand, model and year of cars from the Stanford Cars Dataset. The idea behind this was to give rise to a scalable model that could be run on different systems which have varying availability of hardware resources such as VRAM, GPUs, CPUs etc. Generated a model with a backbone (EfficientNet-B3) that trained 19 million parameters on a local laptop machine, providing an accuracy of 86%. To further demonstrate the scalability, modified the model with a higher end backbone (EfficientNet-B5) that trained 37 million parameters on Google Colab server with premium GPUs to provide a model with accuracy of 90.9%.

 ![alt text](https://github.com/roniepatil/CarNet-Deep-learning-for-car-classification/blob/main/image_resources/output.png)

