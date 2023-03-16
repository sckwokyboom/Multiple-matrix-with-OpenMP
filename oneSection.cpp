#include <iostream>
#include <omp.h>
#include <cmath>

static constexpr int N = 20000;
static constexpr double tau = 0.0001;
static constexpr double eps = 0.000001;
static constexpr int PROGRAM_LAUNCHES_NUM = 20;


double *makeMatrixA() {
    auto *matrix = new double[N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                matrix[i * N + j] = 2.0;
            } else {
                matrix[i * N + j] = 1.0;
            }

        }
    }

    return matrix;
}

double *makeVector(double value) {
    auto *vector = new double[N];
    for (int i = 0; i < N; ++i) {
        vector[i] = (value);
    }
    return vector;
}

void printVector(double *vector) {
    std::cout << "( ";

    for (int i = 0; i < N; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << ")";
}


int main() {
    for (int k = 1; k < PROGRAM_LAUNCHES_NUM; ++k) {
        auto *matrixA = makeMatrixA();
        auto *vectorB = makeVector(N + 1.0);
        auto *vectorX = makeVector(0);
        auto *vectorXBuff = new double[N]();

        double normB = 0;
        double normV = 0;
        double newNormCriteria = 0;
        bool criteria = false;

        omp_set_dynamic(0);
        omp_set_num_threads(k);
        double startTime = omp_get_wtime();

#pragma omp parallel
        {
#pragma omp for schedule(static) reduction(+:normB)
            for (int i = 0; i < N; ++i) {
                normB += vectorB[i] * vectorB[i];
            }

#pragma omp single
            {
                normB = sqrt(normB);
            }

            while (!criteria) {
#pragma omp for schedule(static) reduction(+:normV)
                for (int i = 0; i < N; ++i) {
                    double valueX = 0;

                    for (int j = 0; j < N; ++j) {
                        valueX += matrixA[i * N + j] * vectorX[j];
                    }

                    valueX -= vectorB[i];

                    vectorXBuff[i] = vectorX[i] - valueX * tau;
                    normV += valueX * valueX;
                }

#pragma omp single
                {
                    std::swap(vectorX, vectorXBuff);
                    normV = sqrt(normV);
                    newNormCriteria = normV / normB;
                    criteria = newNormCriteria < eps;
                    normV = 0;
                }
            }
        }

        double end_time = omp_get_wtime();

        std::cout << "Time passed: " << end_time - startTime << " seconds. Threads used: " << omp_get_max_threads()
                  << std::endl;

        delete[] matrixA;
        delete[] vectorX;
        delete[] vectorB;
        delete[] vectorXBuff;
    }
    return 0;
}