#include <sys/time.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

using namespace std;


const int N = 4096;
const int M = 128;
const int blockM[5] = { 8, 16, 32, 64, 128 };
static vector<double> vecBaseTime;
static vector<double> vecSIMDTime;
static vector<double> vecEigenTime;
static vector<double> vecThreadTime4;

static struct timeval startTime;
static struct timeval endTime;
static unsigned long diff;


#define FLOAT_SIZE 8 // 一个AVX向量中包含8个单精度浮点数
#define BYTE_SIZE 16 

const int THREAD_NUMS = 24;
static float RESULT = 0;

float rand_float(float s) {
    return 4 * s * (1 - s);
}

void matrix_gen(float* a, float* b, float* c, float seed) {
    float s = seed;
	//omp_set_num_threads(THREAD_NUMS);
	//#pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        s = rand_float(s);
        a[i] = s;
        s = rand_float(s);
        b[i] = s;
    }
}

// 一维数组的矩阵乘法--串行
void matrix_multiply(float* A, float* B, int n, float* C) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[n * i + j] += A[n * i + k] * B[n * k + j];
            }
        }
    }
}

// 二维数组矩阵乘法--串行
void matrix(float** a, float** b, float** c, int M) {
	int i, j, k;
	for (i = 0; i < M; i++) {
		for (k = 0; k < M; k++) {
			for (j = 0; j < M; j = j + 1) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
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
inline void matrix_mul_bk(int n, int block_size, float* A, float* B, float* C) {
	for (int sj = 0; sj < n; sj += block_size) {
		for (int si = 0; si < n; si += block_size) {
			for (int sk = 0; sk < n; sk += block_size) {
				// base_block_multiply(n, block_size, A + si * n + sk, \
					B + sk * n + sj, C + si * n + sj);
				int i, j, k;
				float* a1 = A + si * n + sk;
				float* b1 = B + sk * n + sj;
				float* c1 = C + si * n + sj;
				for (i = 0; i < block_size; ++i) {
					for (j = 0; j < block_size; ++j) {
						for (k = 0; k < block_size; ++k) {
							c1[i * n + j] += a1[i * n + k] * b1[k * n + j];
						}
					}
				}
			}	
		}			
	}	
}

// 矩阵相乘（分块）--AVX
void matrix_mul_bk_avx(int n, int block_size, float* A, float* B, float* C) {
	for (int sj = 0; sj < n; sj += block_size) {
		for (int si = 0; si < n; si += block_size) {
			for (int sk = 0; sk < n; sk += block_size) {
				int i, j;
				float* a1 = A + si * n + sk;  // 当前的分块矩阵a1
				float* b1 = B + sk * n + sj;
				float* c1 = C + si * n + sj;
				// 为使用SIMD指令，对b1进行转置
				float* b1_t = (float*)_mm_malloc(sizeof(float) * block_size * block_size, 64);
				for (int ti = 0; ti < block_size; ++ti) {
					for (int tj = 0; tj < block_size; ++tj) {
						b1_t[tj * block_size + ti] = b1[ti * block_size + tj];
					}
				}
				for (i = 0; i < block_size; ++i) {
					for (j = 0; j < block_size; ++j) {
						//加载数组当前元素的寄存器，每次读取8个单精度浮点数
						__m256 va1 = _mm256_setzero_ps();
						__m256 vb1 = _mm256_setzero_ps();
						__m256 vc1 = _mm256_setzero_ps();
						__m256 sum_vc = _mm256_setzero_ps();
						for (int m = 0; m < block_size; m += FLOAT_SIZE) {
							va1 = _mm256_load_ps(&a1[i * block_size + m]);
							vb1 = _mm256_load_ps(&b1_t[j * block_size + m]);
							//sum_vc = _mm256_fmadd_ps(va1, vb1, sum_vc);
							sum_vc = _mm256_add_ps(_mm256_mul_ps(va1, vb1), sum_vc);
						}			
						// 累加求和
						sum_vc = _mm256_add_ps(sum_vc, _mm256_permute2f128_ps(sum_vc, sum_vc, 1));
						sum_vc = _mm256_hadd_ps(sum_vc, sum_vc);
						c1[i * n + j] = _mm256_cvtss_f32(_mm256_hadd_ps(sum_vc, sum_vc));
					}
				}
				// 释放_mm_malloc分配的内存
				_mm_free(b1_t);
			}
		}
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

void matrix_mul_bk_avx_multhread(int block_size, float* A, float* B, float* C)
{
	omp_set_num_threads(THREAD_NUMS);
	#pragma omp parallel for
	for (int sj = 0; sj < N; sj += block_size) {
		for (int si = 0; si < N; si += block_size) {
			for (int sk = 0; sk < N; sk += block_size) {
				// for (int i = sj; i < sj+block_size && i < N; ++i) {
				// 	for (int k = sk; k < sk + block_size && k < N; ++k){
				// 		__m256 a_val = _mm256_broadcast_ss(A + k + i * N);
				// 		for (int j = si; j < si + block_size && j < N; j += 8) {
				// 			__m256 b_val = _mm256_load_ps(B + j + k * N);
				// 			__m256 c_val = _mm256_load_ps(C + j + i * N);
				// 			c_val = _mm256_add_ps(c_val, _mm256_mul_ps(b_val, a_val));
				// 			_mm256_store_ps(C + j + i * N, c_val);
				// 		}
				// 	}
				// }

				int i, j;
				float* a1 = A + si * N + sk;  // 当前的分块矩阵a1
				float* b1 = B + sk * N + sj;
				float* c1 = C + si * N + sj;			
				// 为使用SIMD指令，对b1进行转置
				float* b1_t = (float*)_mm_malloc(sizeof(float) * block_size * block_size, 64);
				for (int ti = 0; ti < block_size; ++ti) {
					for (int tj = 0; tj < block_size; ++tj) {
						b1_t[tj * block_size + ti] = b1[ti * block_size + tj];
					}
				}
				for (i = 0; i < block_size; ++i) {
					for (j = 0; j < block_size; ++j) {
						//加载数组当前元素的寄存器，每次读取8个单精度浮点数
						__m256 va1 = _mm256_setzero_ps();
						__m256 vb1 = _mm256_setzero_ps();
						__m256 vc1 = _mm256_setzero_ps();
						__m256 sum_vc = _mm256_setzero_ps();
						for (int m = 0; m < block_size; m += FLOAT_SIZE) {
							va1 = _mm256_load_ps(&a1[i * block_size + m]);
							vb1 = _mm256_load_ps(&b1_t[j * block_size + m]);
							//sum_vc = _mm256_fmadd_ps(va1, vb1, sum_vc);
							sum_vc = _mm256_add_ps(_mm256_mul_ps(va1, vb1), sum_vc);
						}			
						// 累加求和
						sum_vc = _mm256_add_ps(sum_vc, _mm256_permute2f128_ps(sum_vc, sum_vc, 1));
						sum_vc = _mm256_hadd_ps(sum_vc, sum_vc);
						c1[i * N + j] = _mm256_cvtss_f32(_mm256_hadd_ps(sum_vc, sum_vc));
					}
				}
				// 释放_mm_malloc分配的内存
				_mm_free(b1_t);
		
			}
		}
	}
	// 找出每一行中最大值的最小值
	vector<float> vecMax(N);
	omp_set_num_threads(4);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		vecMax[i] = findMax(C+i*N);
	}
	
	RESULT = findMin(vecMax);
}

inline void simd_omp_block(float *c, float *a, float *b, int m0)
{
    #pragma omp parallel for
    for(int i = 0; i < N; i += m0) {
        for(int j = 0; j < N; j += m0) {
            for(int k = 0; k < N; k += m0) {
                for(int i1 = i; i1 < i + m0 && i1 < N; i1++) {
                    for(int k1 = k; k1 < k + m0 && k1 < N; k1++) {
                        __m256 a_elem = _mm256_broadcast_ss(a + k1 + i1 * N);

                        for(int j1 = j; j1 < j + m0 && j1 < N; j1+=8) {
							 __m256 b_row = _mm256_load_ps(b + j1 + k1 * N); 
    						 __m256 c_row = _mm256_load_ps(c + j1 + i1 * N);

                            c_row = _mm256_add_ps(c_row, _mm256_mul_ps(b_row, a_elem));
							_mm256_store_ps(c + j1 + i1  * N, c_row);
                        }
                    }
                }
            }
        }
    }
	// 找出每一行中最大值的最小值
	vector<float> vecMax(N);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		vecMax[i] = findMax(c+i*N);
	}
	
	RESULT = findMin(vecMax);
}

void run_solution_simd(int n, int m, float rseed)
{
	// 在堆中存放矩阵
	float* A = (float*)_mm_malloc(sizeof(float) * n * n, 64); // 按64字节对齐
	float* B = (float*)_mm_malloc(sizeof(float) * n * n, 64);
	float* C = (float*)_mm_malloc(sizeof(float) * n * n, 64);

	// 随机生成矩阵A和B
	matrix_gen(A, B, C, rseed);

	gettimeofday(&startTime, NULL);

	// 矩阵相乘
	matrix_mul_bk_avx(n, m, A, B, C);

	gettimeofday(&endTime, NULL);
	diff = 1000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000;
	vecSIMDTime.push_back(diff);
	// 释放
	_mm_free(A);
	_mm_free(B);
	_mm_free(C);
}

void run_solution_base(int n, int m, float rseed)
{
	// 在堆中存放矩阵
	float* A = new float[ n * n ];
	float* B = new float[ n * n ];
	float* C = new float[ n * n ];

	// 随机生成矩阵A和B
	matrix_gen(A, B, C, rseed);

	gettimeofday(&startTime, NULL);
	// 矩阵相乘
	matrix_mul_bk(n, m, A, B, C);

	gettimeofday(&endTime, NULL);
	diff = 1000000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec);
	// 输出计时结果，以微秒为单位
	cout << "N = " << n << ", M = " << m << ", time cost: " << diff << " us" << endl;

	vecBaseTime.push_back(diff);
	// 释放内存
	delete[] A;
	delete[] B;
	delete[] C;
}

void run_solution_simd_thread(int m, float rseed)
{
	// 在堆中存放矩阵
	float* A = (float*)_mm_malloc(sizeof(float) * N * N, 64); // 按64字节对齐
	float* B = (float*)_mm_malloc(sizeof(float) * N * N, 64);
	float* C = (float*)_mm_malloc(sizeof(float) * N * N, 64);

	// 随机生成矩阵A和B
	matrix_gen(A, B, C, rseed);

	gettimeofday(&startTime, NULL);

	// 矩阵相乘
	//simd_omp_block(C,A,B,M);
	matrix_mul_bk_avx_multhread(m, A, B, C);

	gettimeofday(&endTime, NULL);
	diff = 1000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000;
	vecThreadTime4.push_back(diff);
	// 释放
	_mm_free(A);
	_mm_free(B);
	_mm_free(C);
}

int main(int argc, char* argv[])
{	
	float rseed = atof(argv[1]);
	// for (int j = 0; j < 5; ++j) {
	// 	run_solution_simd_thread(blockM[j], rseed);	
	// }
	run_solution_simd_thread(M, rseed);	

	int index = 0;
	cout << "Thread num: " << THREAD_NUMS << " 随机数种子：" << rseed << endl;
	// 输出计时结果，以毫秒为单位
	cout << "N = 4096" << ", M = " << M << " time cost: "
		<< vecThreadTime4[0] << " ms"  << endl;
	cout <<" 最大值的最小值是： "<< RESULT << endl;
    return 0;
}
