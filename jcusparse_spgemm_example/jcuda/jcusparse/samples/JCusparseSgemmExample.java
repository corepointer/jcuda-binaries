package jcuda.jcusparse.samples;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseCreateCsr;
import static jcuda.jcusparse.JCusparse.cusparseCsrSetPointers;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseDestroySpMat;
import static jcuda.jcusparse.JCusparse.cusparseSpGEMM_compute;
import static jcuda.jcusparse.JCusparse.cusparseSpGEMM_copy;
import static jcuda.jcusparse.JCusparse.cusparseSpGEMM_createDescr;
import static jcuda.jcusparse.JCusparse.cusparseSpGEMM_destroyDescr;
import static jcuda.jcusparse.JCusparse.cusparseSpGEMM_workEstimation;
import static jcuda.jcusparse.JCusparse.cusparseSpMatGetSize;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseIndexType.CUSPARSE_INDEX_32I;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseSpGEMMDescr;
import jcuda.jcusparse.cusparseSpMatDescr;
import jcuda.runtime.JCuda;

// Ported from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm/spgemm_example.c
// For https://forum.byte-welt.net/t/on-entry-to-cusparsespgemm-createdescr-parameter-number-1-descr-had-an-illegal-value-null-pointer/23472
public class JCusparseSgemmExample
{
    private static final int CUSPARSE_SPGEMM_DEFAULT = 0;

    public static void main(String args[])
    {
        JCuda.setExceptionsEnabled(true);
        JCusparse.setExceptionsEnabled(true);
        
        int A_NUM_ROWS = 4;
        int A_num_rows = 4;
        int A_num_cols = 4;
        int A_nnz      = 9;
        int B_num_rows = 4;
        int B_num_cols = 4;
        int B_nnz      = 9;
        int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
        int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
        float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f };
        int   hB_csrOffsets[] = { 0, 2, 4, 7, 8 };
        int   hB_columns[]    = { 0, 3, 1, 3, 0, 1, 2, 1 };
        float hB_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f };
        int   hC_csrOffsets[] = { 0, 4, 6, 10, 12 };
        int   hC_columns[]    = { 0, 1, 2, 3, 1, 3, 0, 1, 2, 3, 1, 3 };
        float hC_values[]     = { 11.0f, 36.0f, 14.0f, 2.0f,  12.0f,
                                  16.0f, 35.0f, 92.0f, 42.0f, 10.0f,
                                  96.0f, 32.0f };
        
        int C_NUM_NNZ                   = 12;
        int C_nnz                       = 12;
        float               alpha       = 1.0f;
        float               beta        = 0.0f;
        int                 opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
        int                 opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
        int                 computeType = CUDA_R_32F;
        //--------------------------------------------------------------------------
        // Device memory management: Allocate and copy A, B
        // int
        Pointer dA_csrOffsets = new Pointer();
        Pointer dA_columns = new Pointer(); 
        Pointer dB_csrOffsets = new Pointer();
        Pointer dB_columns = new Pointer();
        Pointer dC_csrOffsets = new Pointer(); 
        Pointer dC_columns = new Pointer();
        // float
        Pointer dA_values = new Pointer();
        Pointer dB_values = new Pointer(); 
        Pointer dC_values = new Pointer();
        
        // allocate A
        cudaMalloc(dA_csrOffsets, (A_num_rows + 1) * Sizeof.INT);
        cudaMalloc(dA_columns, A_nnz * Sizeof.INT);
        cudaMalloc(dA_values,  A_nnz * Sizeof.FLOAT);
        // allocate B
        cudaMalloc(dB_csrOffsets, (B_num_rows + 1) * Sizeof.INT);
        cudaMalloc(dB_columns, B_nnz * Sizeof.INT);
        cudaMalloc(dB_values,  B_nnz * Sizeof.FLOAT);
        // allocate C offsets
        cudaMalloc(dC_csrOffsets, (A_num_rows + 1) * Sizeof.INT);

        // copy A
        cudaMemcpy(dA_csrOffsets, Pointer.to(hA_csrOffsets),
            (A_num_rows + 1) * Sizeof.INT, cudaMemcpyHostToDevice);
        cudaMemcpy(dA_columns, Pointer.to(hA_columns), 
            A_nnz * Sizeof.INT, cudaMemcpyHostToDevice);
        cudaMemcpy(dA_values, Pointer.to(hA_values),
            A_nnz * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        // copy B
        cudaMemcpy(dB_csrOffsets, Pointer.to(hB_csrOffsets),
            (B_num_rows + 1) * Sizeof.INT, cudaMemcpyHostToDevice);
        cudaMemcpy(dB_columns, Pointer.to(hB_columns), 
            B_nnz * Sizeof.INT, cudaMemcpyHostToDevice);
        cudaMemcpy(dB_values, Pointer.to(hB_values),
            B_nnz * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseHandle     handle = new cusparseHandle();
        cusparseSpMatDescr matA = new cusparseSpMatDescr();
        cusparseSpMatDescr matB = new cusparseSpMatDescr();
        cusparseSpMatDescr matC = new cusparseSpMatDescr();
        Pointer dBuffer1 = new Pointer();
        Pointer dBuffer2 = new Pointer();
        long aBufferSize1[] = { 0 };
        long aBufferSize2[] = { 0 };
        cusparseCreate(handle);
        // Create sparse matrix A in CSR format
        cusparseCreateCsr(matA, A_num_rows, A_num_cols, A_nnz,
            dA_csrOffsets, dA_columns, dA_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateCsr(matB, B_num_rows, B_num_cols, B_nnz,
            dB_csrOffsets, dB_columns, dB_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateCsr(matC, A_num_rows, B_num_cols, 0,
            null, null, null,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        //--------------------------------------------------------------------------
        // SpGEMM Computation
        cusparseSpGEMMDescr spgemmDesc = new cusparseSpGEMMDescr(); 
        cusparseSpGEMM_createDescr(spgemmDesc);

        Pointer pAlpha = Pointer.to(new float[] { alpha });
        Pointer pBeta = Pointer.to(new float[] { beta });
        
        // ask bufferSize1 bytes for external memory
        cusparseSpGEMM_workEstimation(handle, opA, opB,
            pAlpha, matA, matB, pBeta, matC,
            computeType, CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc, aBufferSize1, new Pointer());
        cudaMalloc(dBuffer1, aBufferSize1[0]);
        // inspect the matrices A and B to understand the memory requirement for
        // the next step
        cusparseSpGEMM_workEstimation(handle, opA, opB,
            pAlpha, matA, matB, pBeta, matC,
            computeType, CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc, aBufferSize1, dBuffer1);

        // ask bufferSize2 bytes for external memory
        cusparseSpGEMM_compute(handle, opA, opB,
            pAlpha, matA, matB, pBeta, matC,
            computeType, CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc, aBufferSize2, new Pointer());
        cudaMalloc(dBuffer2, aBufferSize2[0]);

        // compute the intermediate product of A * B
        cusparseSpGEMM_compute(handle, opA, opB,
            pAlpha, matA, matB, pBeta, matC,
            computeType, CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc, aBufferSize2, dBuffer2);
        // get matrix C non-zero entries C_nnz1
        long aC_num_rows1[] = { 0 };
        long aC_num_cols1[] = { 0 }; 
        long aC_nnz1[] = { 0 };
        cusparseSpMatGetSize(matC, aC_num_rows1, aC_num_cols1, aC_nnz1);
        
        long C_num_rows1 = aC_num_rows1[0];
        long C_num_cols1 = aC_num_cols1[0];
        long C_nnz1 = aC_nnz1[0];
        
        // allocate matrix C
        cudaMalloc(dC_columns, C_nnz1 * Sizeof.INT);
        cudaMalloc(dC_values,  C_nnz1 * Sizeof.FLOAT);

        // NOTE: if 'beta' != 0, the values of C must be update after the allocation
        //       of dC_values, and before the call of cusparseSpGEMM_copy

        // update matC with the new pointers
        
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

        // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

        // copy the final products to the matrix C
        cusparseSpGEMM_copy(handle, opA, opB,
            pAlpha, matA, matB, pBeta, matC,
            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

        // destroy matrix/vector descriptors
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroySpMat(matA);
        cusparseDestroySpMat(matB);
        cusparseDestroySpMat(matC);
        cusparseDestroy(handle);
        
        //--------------------------------------------------------------------------
        // device result check
        int   hC_csrOffsets_tmp[] = new int[A_NUM_ROWS + 1];
        int   hC_columns_tmp[] = new int[C_NUM_NNZ];
        float hC_values_tmp[] = new float[C_NUM_NNZ];
        cudaMemcpy(Pointer.to(hC_csrOffsets_tmp), dC_csrOffsets,
            (A_num_rows + 1) * Sizeof.INT, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(hC_columns_tmp), dC_columns, 
            C_nnz * Sizeof.INT, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(hC_values_tmp), dC_values, 
            C_nnz * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        int correct = 1;
        for (int i = 0; i < A_num_rows + 1; i++) {
            if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i]) {
                correct = 0;
                break;
            }
        }
        for (int i = 0; i < C_nnz; i++) {
            if (hC_columns_tmp[i] != hC_columns[i] ||
                hC_values_tmp[i]  != hC_values[i]) { // direct floating point
                correct = 0;                         // comparison is not reliable
                break;
            }
        }
        if (correct == 1)
            System.out.printf("spgemm_example test PASSED\n");
        else {
            System.out.printf("spgemm_example test FAILED: wrong result\n");
        }
        //--------------------------------------------------------------------------
        // device memory deallocation
        cudaFree(dBuffer1);
        cudaFree(dBuffer2);
        cudaFree(dA_csrOffsets);
        cudaFree(dA_columns);
        cudaFree(dA_values);
        cudaFree(dB_csrOffsets);
        cudaFree(dB_columns);
        cudaFree(dB_values);
        cudaFree(dC_csrOffsets);
        cudaFree(dC_columns);
        cudaFree(dC_values);
    }
}
