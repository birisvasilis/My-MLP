-A CUSTOMIZABLE MULTI-LAYER PERCEPTRON (MLP) IN C++

The goal of this project is to provide a black-box like, fully customizable MLP for machine learning in C++. In order to use it, simply include the MLP.hpp header file. The fast and open-source 
Eigen Matrix library is also needed. You can see more on Eigen here: https://github.com/PX4/eigen

-CONSTRUCTING THE MLP

Calling the constructor involves passing an integer that represents the depth of the MLP, followed by an array of ints, with the same size as the first integer parameter, that represent the heights of each of the layers.

-TRAINING AND TESTING THE MLP

In order to train and test the MLP, the data needs to be in the form of four STL vectors, of type eigen::VectorXd. Two of the vectors contain the training and testing sets, and the other 
two contain the accompanying labels to each of the training and testing sets.

Once the data is formated correctly, training the MLP is as simple as calling the train() member function, passing the training set and training labels vectors, as well
as the learning rate parameter that is used during the stochastic gradient descent. Testing the MLP is done by using the test() member function, passing the testing set and corresponing
labels vectors. Both functions provide percentage-based progress updates on the training and testing procedures, while the test() function provides the exact number of 
correct choices and the accuracy of the MLP on the given testing set.

<img width="398" height="70" alt="image" src="https://github.com/user-attachments/assets/74c8b38d-ecd7-4da7-bf49-11994ae27736" />




It is also possible to execute a single forward pass on some given input data of type Eigen::VectorXd, using the forward_pass() member function, which returns an Eigen::VectorXd
consisting of the activations of the last layer. In the case where the choice of the MLP is translated as the last layer node with the highest activation, the choice() member
function is provided. By passing the output of forward_pass() into choice(), choice() returns an integer that represents the index of the node with the highest activation at the
last layer, i.e the "choice" of the MLP.

-ABOUT

This is an ongoing project for which more features are planned, such as batch-training, epochs, more performance metrics and more.
