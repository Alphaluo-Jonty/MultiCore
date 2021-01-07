#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


static int N = 4096;
static float rseed = 0.925;


static struct timeval startTime;
static struct timeval endTime;
static unsigned long diff;



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

void run_eigen()
{
	// 在堆中存放矩阵,用MatrixXf类型
	MatrixXf A(N, N);
	MatrixXf B(N, N);
	MatrixXf C(N, N);

	// 随机生成矩阵A和B
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A(i, j) = rand_float(rseed);
			B(i, j) = rand_float(rseed);
			C(i, j) = 0;
		}
	}


	gettimeofday(&startTime, NULL);

	C = A * B;

	gettimeofday(&endTime, NULL);
	diff = 1000 * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000;
	
}

int main(int argc, char* argv[])
{	

	if (argv[1] != nullptr)
		N = atof(argv[1]);

	if (argv[2] != nullptr)
		rseed = atof(argv[2]);

	
	run_eigen();	

	cout << "N =" << N << " 随机数种子：" << rseed << endl;
	cout << "eigen time cost: "
		<< diff << " ms"  << endl;
	
    return 0;
}
