#include "MLP.hpp"
#include "mnist_reader.hpp"  
#include <ctime>

int main(){
    std::srand((unsigned int) std::time(nullptr)); // seeding the random generator

    const std::string path = "C:/Users/biris/.vscode/cli/data/";
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);
    
    //the goal now is to parse the input and target data into eigen vectors:
    VectorXd image(784);
    std::vector<VectorXd> training_set;
    auto it1 = dataset.training_images.begin();
    
    while( it1 != dataset.training_images.end() ) {
        auto it2 = it1->begin();
        int index = 0;
        while(it2 != it1->end()){
            image[index] = (*it2 / 255.0);
            index++;
            it2++;
        }
        training_set.push_back(image);
        it1++;
    }

    std::vector<VectorXd> testing_set;
    it1 = dataset.test_images.begin();
    
    while( it1 != dataset.test_images.end() ) {
        auto it2 = it1->begin();
        int index = 0;
        while(it2 != it1->end()){
            image[index] = (*it2 / 255.0);
            index++;
            it2++;
        }
        testing_set.push_back(image);
        it1++;
    }

    //now for the target labels, as one-hot encoded vectors of the correct digit
    std::vector<VectorXd> training_labels;
    auto it3 = dataset.training_labels.begin();
        while(it3 != dataset.training_labels.end()){
            VectorXd label = VectorXd::Zero(10);
            label[*it3] = 1.0; // if answer is 3, then everything will be zero except of index no.3, which will be 1.0
            training_labels.push_back(label);
            it3++;
        }
    
    std::vector<VectorXd> testing_labels;
    it3 = dataset.test_labels.begin();
        while(it3 != dataset.test_labels.end()){
            VectorXd label = VectorXd::Zero(10);
            label[*it3] = 1.0; // if answer is 3, then everything will be zero except of index no.3, which will be 1.0
            testing_labels.push_back(label);
            it3++;
        }

        
    /*std::cout << "Training images loaded: " << dataset.training_images.size() << std::endl;
    std::cout << "Training labels loaded: " << dataset.training_labels.size() << std::endl;
    std::cout << "Test images loaded:     " << dataset.test_images.size() << std::endl; */

    /*once the data is parsed into stl vectors of eigen vectors, the process of 
    training and testing is as simple as this : */
    int size[] = {784, 12, 12, 10}; // suitable MLP for the MNIST database
    MLP mymlp(4, size);
    mymlp.test(testing_set, testing_labels);
    mymlp.train(training_set, training_labels, 0.05);
    mymlp.test(testing_set, testing_labels);
}