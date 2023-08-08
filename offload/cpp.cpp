// C++, initialize and array, do some computation on array and print output

#include <iostream>

int main(){
  // initialize an array
  const int N = 1024;
  float data[N];
  for (int i=0; i<N; i++) data[i] = i;

  // compute on CPU
  for (int i=0; i<N; i++) data[i] *= 5;

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";
}
