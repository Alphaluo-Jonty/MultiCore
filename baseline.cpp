#include <sys/time.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;


static int N = 4096;
static int M = 128;
static float rseed = 0.925;
static float RESULT = 0;


static vector<double> vecBaseTime;

static struct timeval startTime;
static struct timeval endTime;
static unsigned long diff;


float rand_float(float s) {
    return 4 * s * (1 - s);
}

void matrix_gen(float* a, float* b, float seed) {
    float s = seed;
    for (int i = 0; i < N * N; ++i) {
        s = rand_float(s);
        a[i] = s;
        s = rand_float(s);
        b[i] = s;
    }
}


inline float findMax(float* array) {
	float max = -1000;
	for (int i = 0; i < N; ++i){
		if (max < array[i]) {
			max = array[i];
		}
	}
	return max;
}

float findMin(vector<float> vecVal) {
	float minVal = 100000;
	for (int i = 0; i < vecVal.size(); ++i) {
		if (minVal > vecVal[i]) 
			minVal = vecVal[i];
	}
	return minVal;
}

// 分块矩阵相乘--串行
inline void base_block_multiply(int n, int block_size, float* A, float* B, float* C) {
	int i, j, k;
	for (i = 0; i < block_size; ++i) {
		for (j = 0; j < block_size; ++j) {
			for (k = 0; k < block_size; ++k) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}


// 矩阵相乘（分块）--串行
inline void mul_block(int block_size, float* A, float* B, float* C) {
	for (int sj = 0; sj < N; sj += block_size) {
		for (int si = 0; si < N; si += block_size) {
			for (int sk = 0; sk < N; sk += block_size) {
				// base_block_multiply(N, block_size, A + si * N + sk, \
					B + sk * N + sj, C + si * N + sj);
				int i, j, k;
				float* a1 = A + si * N + sk;
				float* b1 = B + sk * N + sj;
				float* c1 = C + si * N + sj;
				for (i = 0; i < block_size; ++i) {
					for (j = 0; j < block_size; ++j) {
						for (k = 0; k < block_size; ++k) {
							c1[i * N + j] += a1[i * N + k] * b1[k * N + j];
						}
					}
				}
			}	
		}			
	}	
	// 找出每一行中最大值的最小值
	vector<float> vecMax(N);
	for (int i = 0; i < N; ++i) {
		vecMax[i] = findMax(C+i*N);
	}
	
	RESULT = findMin(vecMax);
}

void run_baseline()
{
	// 在堆中存放矩阵
	float* A = new float[ N * N ];
	float* B = new float[ N * N ];
	float* C = new float[ N * N ];

	// 随机生成矩阵A和B
	matrix_gen(A, B, rseed);

	gettimeofday(&startTime, NULL);

	// 矩阵相乘
	mul_block(M, C, A, B);

	gettimeofday(&endTime, NULL);
	diff = 1000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000;
	vecBaseTime.push_back(diff);

	// 释放内存
	delete[] A;
	delete[] B;
	delete[] C;
}

int main(int argc, char* argv[])
{	
	if (argv[1] != nullptr)
		N = atof(argv[1]);

	if (argv[2] != nullptr)
		rseed = atof(argv[2]);

	
	run_baseline();	

	cout << "N = " << N << " 随机数种子：" << rseed << endl;
	cout <<" 最大值的最小值是： "<< RESULT << endl;
	cout << "M = " << M << " time cost: "
		<< vecBaseTime[0] << " ms"  << endl;
	
	// M = 64;
	// run_baseline();		
	// cout << "M = " << M << " time cost: "
	// 	<< vecBaseTime[1] << " ms"  << endl;
	
	// M = 32;
	// run_baseline();	
	// cout << "M = " << M << " time cost: "
	// 	<< vecBaseTime[2] << " ms"  << endl;
	
	// M = 16;
	// run_baseline();	
	// cout << "M = " << M << " time cost: "
	// 	<< vecBaseTime[3] << " ms"  << endl;
	
	// M = 8;
	// run_baseline();	
	// cout << "M = " << M << " time cost: "
	// 	<< vecBaseTime[4] << " ms"  << endl;
	
    return 0;
}
