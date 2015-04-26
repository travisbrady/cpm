#include <stdio.h>
#include "cpm_c.h"

extern "C" {
    cpm_model create() {
        CPM* model = nullptr;
        model = new CPM(10, 1, 1.0, 0.0, 1.0, (unsigned short) 42);
        return (cpm_model)model;
    }

    //StochasticDataAdaptor(float* data, int* labels, size_t n_instances, size_t n_dimensions);
    //void fit(const StochasticDataAdaptor& trainset, int iterations, bool reshuffle, bool verbose);
    void fit(cpm_model c_model, float* data, int* labels, size_t n_instances, size_t n_dimensions, int iterations, bool reshuffle, bool verbose) {
        printf("sizeof(int) %lu\n", sizeof(int));
        printf("sizeof(float) %lu\n", sizeof(float));
        CPM* model = (CPM*)c_model;
        StochasticDataAdaptor sda = StochasticDataAdaptor(data, labels, n_instances, n_dimensions);
        model->fit(sda, iterations, reshuffle, verbose);
    }
    
    //void predict(const StochasticDataAdaptor& testset, float* scores, int* assignments) const;
    void predict(cpm_model c_model, float* data, int* labels, size_t n_instances, size_t n_dimensions, float* scores, int* assignments) {
        printf("[predict] assignments: %d %d\n", assignments[0], assignments[1]);
        printf("[predict] labels: %d %d\n", labels[0], labels[1]);
        CPM* model = (CPM*)c_model;
        StochasticDataAdaptor sda = StochasticDataAdaptor(data, labels, n_instances, n_dimensions);
        model->predict(sda, scores, assignments);
        printf("[predict] assignments: %d %d\n", assignments[0], assignments[1]);
        printf("[predict] labels: %d %d\n", labels[0], labels[1]);
    }

    void serializeModel(cpm_model c_model, const char* filename) {
        CPM* model = (CPM*)c_model;
        model->serializeModel(filename);
    }

    int get_outer_label(cpm_model c_model) {
        CPM* model = (CPM*)c_model;
        return model->outer_label;
    }

    int get_k(cpm_model c_model) {
        CPM* model = (CPM*)c_model;
        return model->k;
    }
}
