#ifndef DOL_TEST_CUBLAS_UTILS_H
#define DOL_TEST_CUBLAS_UTILS_H

#include <cstdlib>
#include "cublas_v2.h"


static void cublasCheckError(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cerr <<"CUBLAS_STATUS_NOT_INITIALIZED\n"; std::exit(1);

        case CUBLAS_STATUS_ALLOC_FAILED:
            std::cerr <<"CUBLAS_STATUS_ALLOC_FAILED\n"; std::exit(1);

        case CUBLAS_STATUS_INVALID_VALUE:
            std::cerr <<"CUBLAS_STATUS_INVALID_VALUE\n"; std::exit(1);

        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cerr <<"CUBLAS_STATUS_ARCH_MISMATCH\n"; std::exit(1);

        case CUBLAS_STATUS_MAPPING_ERROR:
            std::cerr <<"CUBLAS_STATUS_MAPPING_ERROR\n"; std::exit(1);

        case CUBLAS_STATUS_EXECUTION_FAILED:
            std::cerr <<"CUBLAS_STATUS_EXECUTION_FAILED\n"; std::exit(1);

        case CUBLAS_STATUS_INTERNAL_ERROR:
            std::cerr <<"CUBLAS_STATUS_INTERNAL_ERROR\n"; std::exit(1);
    }

}

#endif //DOL_TEST_CUBLAS_UTILS_H