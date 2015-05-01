#include "cpm.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void * cpm_model;
    typedef void * cpm_dataset;
    cpm_model create(void);
    void fit(cpm_model, float*, int*, size_t, size_t, int, bool, bool);
    void predict(cpm_model, float*, int*, size_t, size_t, float*, int*);
    void predict_dataset(cpm_model, cpm_dataset, float*, int*);
    void serializeModel(cpm_model, const char*);
    cpm_model deserializeModel(const char* filename);
    int get_outer_label(cpm_model);
    int get_k(cpm_model);
    cpm_dataset cpm_dataset_from_file(const char*, size_t);
    cpm_dataset cpm_dataset_from_dense_memory(float*, int*, size_t, size_t);
    size_t cpm_dataset_get_n_instances(cpm_dataset);

#ifdef __cplusplus
}
#endif
