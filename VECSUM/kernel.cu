#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include <iomanip>
#include <chrono>

__global__
void vectorSum(int* d_vector, int* d_sum, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(d_sum, d_vector[tid]);
    }
}

int cpu_sum(int* arr, int N) {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    // Размер вектора
    int size = 1000000;

    // Выделяем и заполняем вектор случайными значениями
    int* h_vector = new int[size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 100);
    for (int i = 0; i < size; ++i) {
        h_vector[i] = dis(gen);
    }

    // Выделяем память на GPU
    int* d_vector;
    cudaMalloc((void**)&d_vector, size * sizeof(int));
    int* d_sum;
    int cpu_sumres;
    clock_t startcpu, endcpu;

    startcpu = clock();
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    cpu_sumres = cpu_sum(h_vector, size);
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

    endcpu = clock();
    std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    double elapsed_seconds = elapsed_time.count();
    double cpu_time_taken = double(endcpu - startcpu) / CLOCKS_PER_SEC;
    std::cout << "Time (CPU): "  << elapsed_seconds << " seconds\n" << "CPU result:" << cpu_sumres << std::endl;
    cudaMalloc((void**)&d_sum, sizeof(int));

    // Копируем вектор с хоста на устройство
    cudaMemcpy(d_vector, h_vector, size * sizeof(int), cudaMemcpyHostToDevice);

    // Вычисляем количество блоков и количество потоков в блоке
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Запускаем ядро CUDA для вычисления суммы элементов вектора
    vectorSum << <blocksPerGrid, threadsPerBlock >> > (d_vector, d_sum, size);
    // Установка точки окончания
    cudaEventRecord(stop, 0);
    // Синхронизация устройств
    cudaDeviceSynchronize();


    // Расчет времени
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("(GPU) Time %s: %.9f seconds\n", ":", gpuTime / 1000);
    // Копируем результат с устройства на хост
    int h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU result: " << h_sum << std::endl;

    // Освобождаем память на GPU и хосте
    cudaFree(d_vector);
    cudaFree(d_sum);
    delete[] h_vector;

    return 0;
}
