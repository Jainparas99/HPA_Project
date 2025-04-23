#include <iostream>
#include "common.h"

bool test_basic_csv_read() {
    auto data = read_csv("config/test_models.csv");

    if (data.size() != 2 || data[0].size() != 2 || data[1].size() != 2) {
        std::cerr << "Unexpected CSV size\n";
        return false;
    }

    return data[0][0] == "a" &&
           data[0][1] == "b" &&
           data[1][0] == "c" &&
           data[1][1] == "d";
}

int main() {
    if (test_basic_csv_read()) {
        std::cout << "✅ Test passed!\n";
        return 0;
    } else {
        std::cerr << "❌ Test failed.\n";
        return 1;
    }
}
