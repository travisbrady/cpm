#include <stdio.h>
#include "cpm_c.h"

extern "C" {
    cpm_model create() {
        CPM* model = nullptr;
        model = new CPM(10, 1, 1.0, 0.0, 1.0, (unsigned short) 42);
        return (cpm_model)model;
    }

    void fit(cpm_model c_model, float* data, int* labels, size_t n_instances, size_t n_dimensions, int iterations, bool reshuffle, bool verbose) {
        CPM* model = (CPM*)c_model;
        StochasticDataAdaptor sda = StochasticDataAdaptor(data, labels, n_instances, n_dimensions);
        model->fit(sda, iterations, reshuffle, verbose);
    }
    
    void predict(cpm_model c_model, float* data, int* labels, size_t n_instances, size_t n_dimensions, float* scores, int* assignments) {
        CPM* model = (CPM*)c_model;
        StochasticDataAdaptor sda = StochasticDataAdaptor(data, labels, n_instances, n_dimensions);
        model->predict(sda, scores, assignments);
    }

    void predict_dataset(cpm_model c_model, cpm_dataset c_dataset, float* scores, int* assignments) {
        CPM* model = (CPM*)c_model;
        StochasticDataAdaptor* sda = static_cast<StochasticDataAdaptor*>(c_dataset);
        model->predict(*sda, scores, assignments);
    }

    void serializeModel(cpm_model c_model, const char* filename) {
        CPM* model = (CPM*)c_model;
        model->serializeModel(filename);
    }

    cpm_model deserializeModel(const char* filename) {
        return (cpm_model)CPM::deserializeModel(filename);
    }

    int get_outer_label(cpm_model c_model) {
        CPM* model = (CPM*)c_model;
        return model->outer_label;
    }

    int get_k(cpm_model c_model) {
        CPM* model = (CPM*)c_model;
        return model->k;
    }

    cpm_dataset cpm_dataset_from_file(const char* fname, size_t n_instances) {
        StochasticDataAdaptor* sda = new StochasticDataAdaptor::StochasticDataAdaptor(fname, n_instances);
        return (cpm_dataset)sda;
    }

    cpm_dataset cpm_dataset_from_dense_memory(float* data, int* labels, size_t n_instances, size_t n_dimensions) {
        StochasticDataAdaptor sda = StochasticDataAdaptor(data, labels, n_instances, n_dimensions);
        StochasticDataAdaptor* p_sda = static_cast<StochasticDataAdaptor*>(&sda);
        return static_cast<cpm_dataset>(p_sda);
    }

    size_t cpm_dataset_get_n_instances(cpm_dataset c_ds) {
        StochasticDataAdaptor* p_ds = (StochasticDataAdaptor*)c_ds;
        printf("Ninst: %lu\n", p_ds->getNInstances());
        return p_ds->getNInstances();
    }
}
