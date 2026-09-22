#ifndef MLP_HPP
#define MLP_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath> 
#include <string>   
#include <exception>
 
using Eigen::VectorXd;
using Eigen::MatrixXd;

//i will view the MLP as a vector of consecutive layers

struct sigmoid{
    double operator()(double x) const{
        return 1.0 / (1.0 + exp(-x));
    }
};

struct sigmoid_prime {
    double operator()(double x) const {
        double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    }
};



class layer {
    
    public:
    layer(int i); // size 
    int get_size();
    MatrixXd get_weights(); // matrix for weights of each node connected to next layer 
    void set_weights(const MatrixXd &m);
    VectorXd get_biases(); // vector for bias of each node
    void set_biases(const VectorXd &d);
    VectorXd get_activations(); // vector for activation of each layer 
    void set_activations(const VectorXd &x);
    VectorXd get_pre_activations(); 
    void set_pre_activations(const VectorXd &x);

    private:
    int layer_size;
    VectorXd pre_activations;
    VectorXd activations;
    VectorXd biases;
    MatrixXd weights;
    
};

inline layer::layer(int i): layer_size(i){
    VectorXd d = VectorXd::Zero(i);
    this->activations = d;
    this->pre_activations = d;
}

inline int layer::get_size(){
    return this->layer_size;
}

inline void layer::set_weights(const MatrixXd &m){
   this->weights = m;
}

inline MatrixXd layer::get_weights(){
    return this->weights;
}

inline void layer::set_biases(const VectorXd &d){
    this->biases = d;
}

inline VectorXd layer::get_biases(){
    return this->biases;
}

 inline void layer::set_activations(const VectorXd &x){
    this->activations = x;
 }

 inline VectorXd layer::get_activations(){
    return this->activations;
 }

  inline void layer::set_pre_activations(const VectorXd &x){
    this->pre_activations = x;
 }

 inline VectorXd layer::get_pre_activations(){
    return this->pre_activations;
 }

class MLP{
    public:
    MLP(int depth, int layers[]);//depth increments by 1 for each layer
   // ~MLP();
    MatrixXd test_weights(int i);
    VectorXd test_biases(int i);
    VectorXd forward_pass(const VectorXd &input);
    VectorXd test_activations(int i);
    void backpropagation(const VectorXd &input, const VectorXd &target, double learning_rate);
    void train(const std::vector<VectorXd> &training_set, const std::vector<VectorXd> &training_labels, double learning_rate);
    int choice(const VectorXd &result); //specific for applications where the model's choice is the largest by value activation on the last layer
    void test(const std::vector<VectorXd> &testing_set, const std::vector<VectorXd> &testing_labels);

    private:
    std::vector<layer> mlp;
 
};


inline MLP::MLP(int depth, int layers[]){

    for(int i = 0; i < depth; i++){
        layer x(layers[i]);
        mlp.push_back(x);
    }

//initializing the weight / bias matrices randomly.
    for(int i = 0; i < depth - 1; i++){//up to the second to last layer
        int fan_in = layers[i];
        int fan_out = layers[i+1];
        double a = std::sqrt(6.0 / (fan_in + fan_out));
        MatrixXd m = MatrixXd::Random(mlp[i+1].get_size(), mlp[i].get_size());
        m = m * a;
        mlp[i].set_weights(m);
    }

    for(int i = 0; i < depth; i ++){
        VectorXd d = VectorXd::Zero(mlp[i].get_size());
        mlp[i].set_biases(d);
    }
    
}

inline VectorXd MLP::forward_pass(const VectorXd &input){//executes a forward pass on the input data vector
    try {
        if(input.size() != mlp[0].get_size())
            throw std::logic_error("Input vector size doesn't match input layer size");
        
        auto current = mlp.begin();
        current->set_activations(input);
        auto next = mlp.begin() + 1;

        while(next != mlp.end()){
            assert(current->get_weights().cols() == current->get_activations().rows());
           /* std::cout << current->get_weights().rows() << ' ' << current->get_weights().cols() << ' '
                      << current->get_activations().rows() << ' ' << current->get_activations().cols()  << std::endl; *///debug print the dimensions 
            VectorXd next_activations = current->get_weights()*current->get_activations() + next->get_biases();
            next->set_pre_activations(next_activations);
            next_activations = next_activations.unaryExpr(sigmoid());
            next->set_activations(next_activations);
            current++;
            next++;
        }

    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return mlp.back().get_activations();
    
}

inline void MLP::backpropagation(const VectorXd &input, const VectorXd &target, double learning_rate){

    int L = mlp.size() - 1;
    VectorXd output = forward_pass(input);
    std::vector<VectorXd> deltas(mlp.size());

    //compute delta for the last layer (L)
    VectorXd sp_output = mlp[L].get_pre_activations().unaryExpr(sigmoid_prime()); //sigma prime of the last pre activations, i.e del alpha to del zeta
    VectorXd delta_L = (output - target).cwiseProduct(sp_output); // chain rule, del E to del alpha times del alpha to del zeta
    deltas[L] = delta_L;
    //propagate the deltas backwards

    for (int i = L - 1; i > 0; --i) {
        VectorXd sp = mlp[i].get_pre_activations().unaryExpr(sigmoid_prime());
        deltas[i] = (mlp[i].get_weights().transpose() * deltas[i + 1]).cwiseProduct(sp);// math xd
}

    for (int i = 0; i < L; ++i) {
    // based on deltas, compute del E del W and del E del b for all weights and biases 
    MatrixXd weight_grad = deltas[i + 1] * mlp[i].get_activations().transpose();
    VectorXd bias_grad = deltas[i + 1];

    // gradient descent
    MatrixXd new_weights = (mlp[i].get_weights() - learning_rate * weight_grad);
    VectorXd new_biases = (mlp[i + 1].get_biases() - learning_rate * bias_grad);
    mlp[i].set_weights(new_weights);
    mlp[i + 1].set_biases(new_biases);
}

 }

inline void MLP::train(const std::vector<VectorXd> &training_set, const std::vector<VectorXd> &training_labels, double learning_rate){
    try {
        if(training_set.size() != training_labels.size())
            throw std::logic_error("label set and data set cardinality doesnt match");

        int set_size = training_set.size();
        int counter = 0;

        auto it1 = training_set.begin();
        auto it2 = training_labels.begin();
        while(it1 != training_set.end()){
            this->backpropagation(*it1, *it2, learning_rate);
            if(counter % 250 == 0){
                double progress = (counter * 100.0) / set_size;
                int i = progress;
                std::cout << "\rTraining progress: " << i << '%' << std::flush;
            }
            it1++;
            it2++;
            counter++;
        }
        std::cout << "\rTraining complete       " << std::endl;
        
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

inline int MLP::choice(const VectorXd &result){//input the last layer of activations, then the model chooses the highest one
    auto it = result.begin();
    double best = result[0];
    int best_index = 0;
    int index = 0;
    while(it != result.end()){
        if(*it >= best){
            best = *it;
            best_index = index;
        }
        it++;
        index++;
    }
    return best_index;
}
 
inline void MLP::test(const std::vector<VectorXd> &testing_set, const std::vector<VectorXd> &testing_labels){
 try {
    if(testing_set.size() != testing_labels.size())
        throw std::logic_error("label set and data set cardinality doesnt match");

    auto it1 = testing_set.begin();
    auto it2 = testing_labels.begin();
    int total = testing_set.size();
    int counter = 0;
    int succesful = 0;
    while(it1 != testing_set.end()){
        auto res = this->forward_pass(*it1);
        int choice = this->choice(res);
        int actual = this->choice(*it2); //choice function can be reused to determine the label from the label vectors
        if(actual == choice)
            succesful++;
        it1++;
        it2++;
        if(counter % 100 == 0){
            double progress = (counter * 100.0) / total;
            int d = progress;
            std::cout << "\rTesting progress: " << d << '%' << std::flush;
        }
        counter++; 
        }
        std::cout << "\r                                   " ;
        double accuracy = (succesful * 100.0) / total;
        std::cout << std::endl << " Testing complete: " << std::endl
                  << "Out of " << total << " test cases, the MLP made the correct decision " 
                  << succesful << " times." << std::endl << "Accuracy: " 
                  << accuracy << '%' << std::endl;

    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }  
}


inline MatrixXd MLP::test_weights(int i){
    return mlp[i].get_weights();
}

inline VectorXd MLP::test_biases(int i){
    return mlp[i].get_biases();
}

inline VectorXd MLP::test_activations(int i){
    return mlp[i].get_activations();
}

#endif 