#include "cpm.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void * cpm_model;
    cpm_model create(void);
    void fit(cpm_model, float*, int*, size_t, size_t, int, bool, bool);
    void predict(cpm_model, float*, int*, size_t, size_t, float*, int*);
    void serializeModel(cpm_model, const char*);
    int get_outer_label(cpm_model);
    int get_k(cpm_model);

#ifdef __cplusplus
}
#endif
