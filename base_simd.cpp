#include <sys/time.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <immintrin.h>

using namespace std;


static int N = 4096;
static int M = 128;
static float rseed = 0.925;
static float RESULT = 0;


static vector<double> vecSIMDTime;

static struct timeval startTime;
static struct timeval endTime;
static unsigned long diff;


#define FLOAT_SIZE 8 // 一个AVX向量中包含8个单精度浮点数


float rand_float(float s) {
    return 4 * s * (1 - s);
}

void matrix_gen(float* a, float* b, float* c, float seed) {
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

inline void simd_block(float *c, float *a, float *b, int m0)
{
    for(int i = 0; i < N; i += m0) {
        for(int j = 0; j < N; j += m0) {
            for(int k = 0; k < N; k += m0) {
                for(int i1 = i; i1 < i + m0 && i1 < N; i1++) {
                    for(int k1 = k; k1 < k + m0 && k1 < N; k1++) {
                        __m256 a_elem = _mm256_broadcast_ss(a + k1 + i1 * N);

                        for(int j1 = j; j1 < j + m0 && j1 < N; j1+=8) {
							 __m256 b_row = _mm256_loadu_ps(b + j1 + k1 * N); 
    						 __m256 c_row = _mm256_loadu_ps(c + j1 + i1 * N);

                            c_row = _mm256_add_ps(c_row, _mm256_mul_ps(b_row, a_elem));
							_mm256_storeu_ps(c + j1 + i1  * N, c_row);
                        }
                    }
                }
            }
        }
    }
	// 找出每一行中最大值的最小值
	vector<float> vecMax(N);
	for (int i = 0; i < N; ++i) {
		vecMax[i] = findMax(c+i*N);
	}
	
	RESULT = findMin(vecMax);
}

void run_base_simd()
{
	// 在堆中存放矩阵
	float* A = new float[ N * N ];
	float* B = new float[ N * N ];
	float* C = new float[ N * N ];

	// 随机生成矩阵A和B
	matrix_gen(A, B, C, rseed);

	gettimeofday(&startTime, NULL);

	// 矩阵相乘
	simd_block(C, A, B, M);

	gettimeofday(&endTime, NULL);
	diff = 1000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000;
	vecSIMDTime.push_back(diff);

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

	
	run_base_simd();	

	cout << "N = " << N << " 随机数种子：" << rseed << endl;
	cout <<" 最大值的最小值是： "<< RESULT << endl;
	cout << "M = " << M << " time cost: "
		<< vecSIMDTime[0] << " ms"  << endl;
	
	M = 64;
	run_base_simd();			
	cout << "M = " << M << " time cost: "
		<< vecSIMDTime[1] << " ms"  << endl;
	
	M = 32;
	run_base_simd();	
	cout << "M = " << M << " time cost: "
		<< vecSIMDTime[2] << " ms"  << endl;
	
	M = 16;
	run_base_simd();	
	cout << "M = " << M << " time cost: "
		<< vecSIMDTime[3] << " ms"  << endl;
	
	M = 8;
	run_base_simd();	
	cout << "M = " << M << " time cost: "
		<< vecSIMDTime[4] << " ms"  << endl;
	
    return 0;
}
