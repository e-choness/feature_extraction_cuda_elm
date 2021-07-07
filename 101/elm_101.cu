#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define BLOCK_DIM 16
#define train_sample 3060
#define test_sample 6084
#define feature 256
#define hidden 1000
#define output_dim 102
#define batchSize 1

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
			system("pause");																									\
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
			system("pause");																					\
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

void invert(double** src, double** dst, int n)
{
	cublasHandle_t handle;
	cublascall(cublasCreate(&handle));

	int *P, *INFO;

	cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));

	int lda = n;

	double **A = (double **)malloc(batchSize * sizeof(double *));
	double **A_d, *A_dflat;

	cudacall(cudaMalloc(&A_d, batchSize * sizeof(double *)));
	cudacall(cudaMalloc(&A_dflat, n*n*batchSize * sizeof(double)));

	A[0] = A_dflat;
	for (int i = 1; i < batchSize; i++)
		A[i] = A[i - 1] + (n*n);

	cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(double *), cudaMemcpyHostToDevice));

	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(A_dflat + (i*n*n), src[i], n*n * sizeof(double), cudaMemcpyHostToDevice));


	cublascall(cublasDgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));


	int INFOh[batchSize];
	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < batchSize; i++)
		if (INFOh[i] != 0)
		{
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	double **C = (double **)malloc(batchSize * sizeof(double *));
	double **C_d, *C_dflat;

	cudacall(cudaMalloc(&C_d, batchSize * sizeof(double *)));
	cudacall(cudaMalloc(&C_dflat, n*n*batchSize * sizeof(double)));
	C[0] = C_dflat;
	for (int i = 1; i < batchSize; i++)
		C[i] = C[i - 1] + (n*n);
	cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(double *), cudaMemcpyHostToDevice));
	cublascall(cublasDgetriBatched(handle, n, (const double **)A_d, lda, P, C_d, lda, INFO, batchSize));

	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < batchSize; i++)
		if (INFOh[i] != 0)
		{
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(dst[i], C_dflat + (i*n*n), n*n * sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(A_d); cudaFree(A_dflat); free(A);
	cudaFree(C_d); cudaFree(C_dflat); free(C);
	cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}


void matrix_inv(double* src, double* dst, int n)
{
	cublasHandle_t handle;
	cublascall(cublasCreate(&handle));

	int *P, *INFO;

	cudacall(cudaMalloc<int>(&P, n * batchSize * sizeof(int)));
	cudacall(cudaMalloc<int>(&INFO, batchSize * sizeof(int)));

	int lda = n;

	double *A[] = { src };
	double** A_d;
	cudacall(cudaMalloc<double*>(&A_d, sizeof(A)));
	cudacall(cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice));

	cublascall(cublasDgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));

	int INFOh = 0;
	cudacall(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh == n)
	{
		fprintf(stderr, "Factorization Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	double* C[] = { dst };
	double** C_d;
	cudacall(cudaMalloc<double*>(&C_d, sizeof(C)));
	cudacall(cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice));

	cublascall(cublasDgetriBatched(handle, n, A_d, lda, P, C_d, lda, INFO, batchSize));

	
	cudacall(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh != 0)
	{
		fprintf(stderr, "Inversion Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	cudaFree(P), cudaFree(INFO), cublasDestroy(handle);
}

__device__ __forceinline__ double sigmoid(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

__global__ void sigmoid_kernel(const double * __restrict__ src,
	double * __restrict__ dst, int datasize)
{
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < datasize; i += stride) {
		dst[i] = sigmoid(src[i]);
		//dst[i] = (1.0 / (1.0 + exp(-src[i])));
	}
}

void get_data(double *data, char *filename) {
	FILE *file;
	int i = 0;
	file = fopen(filename, "r");
	if (file == NULL) {
		printf("failed to open file\n");
	}

	while (fscanf(file, "%lf", &data[i++]) == 1) {
		fscanf(file, ",");
	}
	fclose(file);
}

void show_data(double *data, int ld, int sd) {
	int i = 0;
	int j = 0;
	for (i = ld-1; i < ld; i++) {
		for (j = 0; j < sd; j++) {
			printf("%f ", data[IDX2C(i, j, ld)]);
		}
		printf("\n");
	}
	printf("\n");
	
}

void argmax(double *data, int *dataout, int ld, int sd) {
	int i = 0;
	int j = 0;
	double temp = 0.0f;

	for (i = 0; i < ld; i++) {
		for (j = 0; j < sd; j++) {
			if (j == 0) {
				temp = data[IDX2C(i, j, ld)];
				dataout[i] = 0;
			}
			else if (temp <= data[IDX2C(i, j, ld)]) {
				temp = data[IDX2C(i, j, ld)];
				dataout[i] = j;
			}
		}
	}
}

void get_accuracy(int *data_true, int *data_pred, int n) {
	int i = 0;
	int count = 0;
	double accuracy = 0.0f;
	for (i = 0; i < n; i++) {
		if (data_true[i] == data_pred[i]) {
			count++;
		}
	}
	printf("The count:%d \n", count);
	accuracy = double(count) / double(n);
	printf("The testing accuracy is:%f \n", accuracy);
}


double get_random()
{
	double upperlimit = 1.0;
	return (((double)rand() / (double)(RAND_MAX)) * upperlimit);
}

//void matmul_validate(double *a, double *b, double *c, int m, int n, int k) {
//	int i = 0;
//	int j = 0;
//	int h = 0;
//	double prod = 0.0f;
//	double error = 0.0f;
//	for (i = 0; i < m; i++) {
//		for (j = 0; j < n; j++) {
//			prod = 0.0f;
//			for (h = 0; h < k; h++) {
//				prod += a[IDX2C(i, h, m)] + b[IDX2C(h, j, k)];
//			}
//			error += prod - c[IDX2C(i, j, m)];
//		}
//	}
//
//	printf("error: %f\n", error);
//}

//__global__ void transpose(double *odata, double *idata, int width, int height)
//{
//	__shared__ double block[BLOCK_DIM][BLOCK_DIM + 1];
//
//	// read the matrix tile into shared memory
//	// load one element per thread from device memory (idata) and store it
//	// in transposed order in block[][]
//	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
//	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
//	if ((xIndex < width) && (yIndex < height))
//	{
//		unsigned int index_in = yIndex * width + xIndex;
//		block[threadIdx.y][threadIdx.x] = idata[index_in];
//	}
//
//	// synchronise to ensure all writes to block[][] have completed
//	__syncthreads();
//
//	// write the transposed matrix tile to global memory (odata) in linear order
//	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
//	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
//	if ((xIndex < height) && (yIndex < width))
//	{
//		unsigned int index_out = yIndex * height + xIndex;
//		odata[index_out] = block[threadIdx.x][threadIdx.y];
//	}
//}

//__global__ void transpose_naive(double *odata, double* idata, int width, int height)
//{
//	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
//	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
//
//	if (xIndex < width && yIndex < height)
//	{
//		unsigned int index_in = xIndex + width * yIndex;
//		unsigned int index_out = yIndex + height * xIndex;
//		odata[index_out] = idata[index_in];
//	}
//}

void x_dot_win(double *win, double *x, char *xtrain_file, char *win_file) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	double *d_xtrain;
	double *d_win;
	double *d_x;
	double alpha = 1.0f;
	double beta = 0.0f;
	double *xtrain = (double*)malloc(train_sample * feature * sizeof(double));
	
	get_data(win, win_file);
	/*double upperlimit = 1.0;
	srand((unsigned)time(NULL));
	for (int i = 0; i < (feature * hidden); i++) {
	win[i] = (((double)rand() / (double)(RAND_MAX)) * upperlimit);
	}*/
	printf("win\n");
	//show_data(win, feature, hidden);
	cudacall(cudaMalloc((void**)&d_xtrain, train_sample * feature * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_win, feature * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_x, train_sample * hidden * sizeof(double)));
	printf("xtrain\n");
	get_data(xtrain, xtrain_file);
	//show_data(xtrain, train_sample, feature);
	cublascall(cublasSetMatrix(train_sample, feature, sizeof(*xtrain), xtrain, train_sample, d_xtrain, train_sample));
	cublascall(cublasSetMatrix(feature, hidden, sizeof(*win), win, feature, d_win, feature));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		train_sample, hidden, feature,
		&alpha, d_xtrain, train_sample,
		d_win, feature,
		&beta, d_x, train_sample));
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cublascall(cublasGetMatrix(train_sample, hidden, sizeof(*x), d_x, train_sample, x, train_sample));
	printf("x\n");
	//matmul_validate(xtrain, win, x, train_sample, hidden, feature);
	//show_data(x, train_sample, hidden);
/*	cublascall(cublasGetMatrix(train_sample, feature, sizeof(*xtrain), d_xtrain, train_sample, xtrain, train_sample));
	printf("xtrain\n");
	show_data(xtrain, train_sample, feature);
	cublascall(cublasGetMatrix(feature, hidden, sizeof(*win), d_win, feature, x, feature));
	printf("win\n");
	show_data(win, feature, hidden)*/;
	
	cublasDestroy(handle);
	cudacall(cudaFree(d_xtrain));
	cudacall(cudaFree(d_win));
	cudacall(cudaFree(d_x));
	free(xtrain);
}

void run_sigmoid(double *x) {
	//show_data(x, train_sample, hidden);
	int size = train_sample * hidden;
	double *d_x;
	dim3 dimBlock(256);
	int threadBlocks = (size + (dimBlock.x - 1)) / dimBlock.x;
	//if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	cudacall(cudaMalloc((void**)&d_x, train_sample * hidden * sizeof(double)));
	cudacall(cudaMemcpy(d_x, x, train_sample * hidden * sizeof(double), cudaMemcpyHostToDevice));
	sigmoid_kernel << <dimGrid, dimBlock >> >(d_x, d_x, size);
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cudacall(cudaMemcpy(x, d_x, train_sample * hidden * sizeof(double), cudaMemcpyDeviceToHost));
	printf("sigmoid\n");

	//show_data(x, train_sample, hidden);
	cudacall(cudaFree(d_x));
}

void lc_plus_xtx(double *xtx, double *lc, int n, double c) {
	int i = 0;
	int j = 0;
	for (i = 0; i < hidden; i++) {
		for (j = 0; j < hidden; j++) {
			if (i == j) {
				lc[IDX2C(i, j, hidden)] = xtx[IDX2C(i, j, hidden)] + c;
			}
			else {
				lc[IDX2C(i, j, hidden)] = xtx[IDX2C(i, j, hidden)] + 0.0f;
			}
		}
	}
}

__global__ void mat_plus(double *x, double c, int N) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int index = col + row * N;
	if (col < N && row < N) {
		x[index] = x[index] + c;
	}
}

//void run_activation(double *x, int ld, int sd) {
//	for (int i = 0; i < (ld*sd); i++) {
//		x[i] = (1.0f / (1.0f + exp(-x[i])));
//	}
//}


void xtx(double *x, double *t) {
	double *d_t;
	double *d_x;
	cublasHandle_t handle;
	cublasCreate(&handle);
	double alpha = 1.0f;
	double beta = 0.0f;
	cudacall(cudaMalloc((void**)&d_t, hidden * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_x, train_sample * hidden * sizeof(double)));
	cublascall(cublasSetMatrix(train_sample, hidden, sizeof(*d_x), x, train_sample, d_x, train_sample));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		hidden, hidden, train_sample,
		&alpha, d_x, train_sample,
		d_x, train_sample,
		&beta, d_t, hidden));
	cublascall(cublasGetMatrix(hidden, hidden, sizeof(*t), d_t, hidden, t, hidden));
	printf("t\n");
	//show_data(t, hidden, hidden);
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cublasDestroy(handle);
	cudacall(cudaFree(d_t));
	cudacall(cudaFree(d_x));
}

void lc_xtx(double *t, double *lc, double c) {
	double alpha = 1.0f;
	double beta = c;
	cublasHandle_t handle;
	cublasCreate(&handle);
	double *d_lc;
	double *d_t;
	cudacall(cudaMalloc((void**)&d_lc, hidden*hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_t, hidden * hidden * sizeof(double)));
	for (int i = 0; i < hidden; i++) {
		for (int j = 0; j < hidden; j++) {
			if (i == j) {
				lc[IDX2C(i, j, hidden)] = 1.0f;
			}
			else {
				lc[IDX2C(i, j, hidden)] = 0.0f;
			}
		}
	}
	printf("lc\n");
	//show_data(lc, hidden, hidden);
	cublascall(cublasSetMatrix(hidden, hidden, sizeof(*t), t, hidden, d_t, hidden));
	cublascall(cublasSetMatrix(hidden, hidden, sizeof(*lc), lc, hidden, d_lc, hidden));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		hidden, hidden, hidden,
		&alpha, d_t, hidden,
		d_lc, hidden,
		&beta, d_lc, hidden));

	cublascall(cublasGetMatrix(hidden, hidden, sizeof(*lc), d_lc, hidden, lc, hidden));
	//cublascall(cublasGetMatrix(hidden, hidden, sizeof(*t), d_t, hidden, t, hidden));
	printf("lc+xtx\n");
	//show_data(lc, hidden, hidden);

	cublasDestroy(handle);
	cudacall(cudaFree(d_t));
	cudacall(cudaFree(d_lc));

}

void psuedo_inverse(double *inv, double *x, double *imp) {
	
	double alpha = 1.0f;
	double beta = 0.0f;
	double *d_imp;
	double *d_inv;
	double *d_x;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudacall(cudaMalloc((void**)&d_inv, hidden * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_x, train_sample * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_imp, hidden * train_sample * sizeof(double)));
	cublascall(cublasSetMatrix(hidden, hidden, sizeof(*inv), inv, hidden, d_inv, hidden));
	cublascall(cublasSetMatrix(train_sample, hidden, sizeof(*x), x, train_sample, d_x, train_sample));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		hidden, train_sample, hidden,
		&alpha, d_inv, hidden,
		d_x, train_sample,
		&beta, d_imp, hidden));
	cublascall(cublasGetMatrix(hidden, train_sample, sizeof(*imp), d_imp, hidden, imp, hidden));
	printf("imp\n");
	//show_data(imp, hidden, train_sample);
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	
	cublasDestroy(handle);
	cudacall(cudaFree(d_inv));
	cudacall(cudaFree(d_x));
	cudacall(cudaFree(d_imp));
}

void hidden_beta(double *imp, double *wout, char *ytrain_file) {
	double alpha = 1.0f;
	double beta = 0.0f;
	double *d_ytrain;
	double *d_imp;
	double *d_wout;
	cublasHandle_t handle;
	cublasCreate(&handle);
	double *ytrain = (double*)malloc(train_sample * output_dim * sizeof(double));

	cudacall(cudaMalloc((void**)&d_imp, hidden * train_sample * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_wout, hidden * output_dim * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_ytrain, train_sample * output_dim * sizeof(double)));
	get_data(ytrain, ytrain_file);
	printf("ytrain\n");
	//show_data(ytrain, train_sample, output_dim);
	cublascall(cublasSetMatrix(hidden, train_sample, sizeof(*imp), imp, hidden, d_imp, hidden));
	//cublascall(cublasSetMatrix(hidden, output_dim, sizeof(*wout), wout, train_sample, d_wout, train_sample));
	cublascall(cublasSetMatrix(train_sample, output_dim, sizeof(*ytrain), ytrain, train_sample, d_ytrain, train_sample));

	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		hidden, output_dim, train_sample,
		&alpha, d_imp, hidden,
		d_ytrain, train_sample,
		&beta, d_wout, hidden));

	cublascall(cublasGetMatrix(hidden, output_dim, sizeof(*wout), d_wout, hidden, wout, hidden));
	printf("wout\n");
	//show_data(wout, hidden, output_dim);

	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());

	cublasDestroy(handle);
	cudacall(cudaFree(d_ytrain));
	cudacall(cudaFree(d_imp));
	cudacall(cudaFree(d_wout));
	free(ytrain);
}

void xt_dot_win(double *win, double *pred1, char *xtest_file) {
	double alpha = 1.0f;
	double beta = 0.0f;
	double *d_xtest;
	double *d_win;
	double *d_pred1;
	cublasHandle_t handle;
	cublasCreate(&handle);
	double *xtest = (double*)malloc(test_sample * feature * sizeof(double));
	cudacall(cudaMalloc((void**)&d_xtest, test_sample * feature * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_win, feature * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_pred1, test_sample * hidden * sizeof(double)));
	get_data(xtest, xtest_file);
	printf("xtest\n");
	//show_data(xtest, test_sample, feature);
	cublascall(cublasSetMatrix(test_sample, feature, sizeof(*xtest), xtest, test_sample, d_xtest, test_sample));
	cublascall(cublasSetMatrix(feature, hidden, sizeof(*win), win, feature, d_win, feature));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		test_sample, hidden, feature,
		&alpha, d_xtest, test_sample,
		d_win, feature,
		&beta, d_pred1, test_sample));
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	
	cublascall(cublasGetMatrix(test_sample, hidden, sizeof(*pred1), d_pred1, test_sample, pred1, test_sample));
	printf("pred1\n");
	//show_data(pred1, test_sample, hidden);
	
	cublasDestroy(handle);
	cudacall(cudaFree(d_win));
	cudacall(cudaFree(d_pred1));
	cudacall(cudaFree(d_xtest));
	free(xtest);

}

void cal_sigmoid(double *x, int size) {
	for (int i = 0; i < size; i++) {
		x[i] = 1.0 / (1.0 + exp(-x[i]));
	}
}

void run_sigmoid_t(double *pred1) {
	double *d_pred1;
	int size = test_sample * hidden;
	dim3 dimBlock(256);
	int threadBlocks = (size+ (dimBlock.x - 1)) / dimBlock.x;
	//if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	cudacall(cudaMalloc((void**)&d_pred1, test_sample * hidden * sizeof(double)));
	cublascall(cublasSetMatrix(test_sample, hidden, sizeof(*pred1), pred1, test_sample, d_pred1, test_sample));

	sigmoid_kernel<<<dimBlock, dimGrid>>>(d_pred1, d_pred1, size);

	cublascall(cublasGetMatrix(test_sample, hidden, sizeof(*pred1), d_pred1, test_sample, pred1, test_sample));
	printf("pred1\n");
	//show_data(pred1, test_sample, hidden);
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cudacall(cudaFree(d_pred1));
}

void activation(double *x, int ld, int sd) {
	int size = ld * sd;
	double *d_x;
	dim3 dimBlock(256);
	int threadBlocks = (size + (dimBlock.x - 1)) / dimBlock.x;
	//if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	cudacall(cudaMalloc((void**)&d_x, ld * sd * sizeof(double)));
	cudacall(cudaMemcpy(d_x, x, ld * sd * sizeof(double), cudaMemcpyHostToDevice));
	sigmoid_kernel << <dimGrid, dimBlock >> >(d_x, d_x, size);
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cudacall(cudaMemcpy(x, d_x, ld * sd * sizeof(double), cudaMemcpyDeviceToHost));
	printf("sigmoid\n");

	//show_data(x, train_sample, hidden);
	cudacall(cudaFree(d_x));
}

void get_ypred(double *pred1, double *wout,double *yp) {
	double alpha = 1.0f;
	double beta = 0.0f;
	double *d_yp;
	double *d_pred1;
	double *d_wout;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudacall(cudaMalloc((void**)&d_yp, test_sample * output_dim * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_pred1, test_sample * hidden * sizeof(double)));
	cudacall(cudaMalloc((void**)&d_wout, hidden * output_dim * sizeof(double)));
	cublascall(cublasSetMatrix(test_sample, hidden, sizeof(*pred1), pred1, test_sample, d_pred1, test_sample));
	cublascall(cublasSetMatrix(hidden, output_dim, sizeof(*wout), wout, hidden, d_wout, hidden));
	cublascall(cublasDgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		test_sample, output_dim, hidden,
		&alpha, d_pred1, test_sample,
		d_wout, hidden,
		&beta, d_yp, test_sample));
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());
	cublascall(cublasGetMatrix(test_sample, output_dim, sizeof(*yp), d_yp, test_sample, yp, test_sample));
	printf("yp\n");
	//show_data(yp, test_sample, output_dim);
	
	cublasDestroy(handle);
	cudacall(cudaFree(d_yp));
	cudacall(cudaFree(d_wout));
	cudacall(cudaFree(d_pred1));
}


int main()
{
	cudacall(cudaSetDevice(0));
	cudacall(cudaDeviceSynchronize());
	cudacall(cudaThreadSynchronize());

	double *ytest, *yp;
	int *ypred, *ytrue;

	char xtrain_file[100] = "xtrain101.csv";
	char xtest_file[100] = "xtest101.csv";
	char ytrain_file[100] = "ytrain101.csv";
	char ytest_file[100] = "ytest101.csv";
	char win_file[100] = "Win101.csv";
	cublasHandle_t handle;

	double c = (1.0f / 100.0f);

	cublasCreate(&handle);

	// x_dot_win
	double *win = (double*)malloc(feature * hidden * sizeof(double));
	double *x = (double*)malloc(train_sample*hidden * sizeof(double));
	x_dot_win(win, x, xtrain_file, win_file);
	

	// sigmoid
	activation(x, train_sample, hidden);
	//int size = train_sample*hidden;
	//cal_sigmoid(x, size);
	//printf("x\n");
	//show_data(x, train_sample, hidden);

	//Xt dot X	
	double *t = (double*)malloc(hidden*hidden * sizeof(double));
	xtx(x, t);
	
	
	//lc + xtx
	double *lc = (double*)malloc(hidden * hidden * sizeof(double));	
	lc_plus_xtx(t, lc, hidden, c);
	//show_data(lc, hidden, hidden);
	
	// inverse
	double **p_inv = (double **)malloc(sizeof(double *));
	double *inv = (double*)malloc(hidden * hidden * sizeof(double));
	p_inv[0] = inv;

	double **p_lc = (double**)malloc(sizeof(double*));
	p_lc[0] = lc;
	invert(p_lc, p_inv, hidden);
	//matrix_inv(lc, inv, hidden);
	printf("inv\n");
	//show_data(inv, hidden, hidden);

	// psuedo inverse
	double *imp = (double*)malloc(hidden* train_sample * sizeof(double));	
	psuedo_inverse(inv, x, imp);
	//show_data(imp, hidden, train_sample);

	// hidden weight beta
	double *wout = (double*)malloc(hidden*output_dim*sizeof(double));
	hidden_beta(imp, wout, ytrain_file);
	//show_data(wout, hidden, output_dim);

	//predict x dot win
	double *pred1 = (double*)malloc(test_sample*hidden * sizeof(double));
	xt_dot_win(win, pred1, xtest_file);
	//show_data(pred1, test_sample, hidden);

	// predict sigmoid
	activation(pred1, test_sample, hidden);
	//show_data(pred1, test_sample, hidden);

	// get predicted result
	yp = (double*)malloc(test_sample*output_dim * sizeof(double));
	get_ypred(pred1, wout, yp);
	//show_data(yp, test_sample, output_dim);

	//get accuracy
	ytest = (double*)malloc(test_sample * output_dim * sizeof(double));
	get_data(ytest, ytest_file);
	printf("ytest\n");
	//show_data(ytest, test_sample, output_dim);

	ypred = (int*)malloc(test_sample * sizeof(int));
	ytrue = (int*)malloc(test_sample * sizeof(int));

	argmax(yp, ypred, test_sample, output_dim);	
	argmax(ytest, ytrue, test_sample, output_dim);
	//printf("ypred\n");
	//show_data(ypred, test_sample, 1);
	/*for (int i = 0; i < test_sample; i++) {
		printf("%d %d\n", ypred[i], ytrue[i]);
	}*/
	get_accuracy(ytrue, ypred, test_sample);
	free(ytest);
	free(yp);
	

	//free spaces
	cublasDestroy(handle);
	free(win);
	free(x);
	free(t);
	free(lc);
	free(p_lc);
	free(inv);
	free(imp);
	free(wout);
	free(pred1);
	free(ypred);
	free(ytrue);

	system("pause");
	return 0;
}