#include "cribArray.h"
#include <iostream>

void printArray(const std::vector<std::vector<float>> arr, int rows, int cols) {

    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << "\n";
    }
    
}

