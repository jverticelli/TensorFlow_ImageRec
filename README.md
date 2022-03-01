# TensorFlow_ImageRec
TensorFlow for Image Recognition.

This guide trains a neural network model to classify images of clothing.
It uses tf.keras, a high-level API to build and train models in TensorFlow.
The Fashion MNIST dataset is used which contains 70.000 grayscale images in 10 categories.
60.000 images are used to train the network and 10.000 images to assess how precisely the network has learned to classify the images.

How to run:
1. Copy the repository: git clone https://github.com/jverticelli/TensorFlow_ImageRec.git
2. Open the project on VSC
3. Create virtual environment: py -m venv myenv
4. Activate virtual environment: myenv\Scripts\activate
5. Upgrade pip: pip install --upgrade pip
6. Install TensorFlow: pip install tensorflow
7. Install matplotlib: pip install matplotlib
8. Run: py App.py
