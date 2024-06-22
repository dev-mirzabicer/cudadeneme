CC := g++
NVCC := nvcc
CXX_FLAGS := -std=c++11 -I.
CUDA_FLAGS := -arch=sm_35

OBJS := matrix.o vector.o

all: myapp test_matrix test_vector

myapp: $(OBJS)
	$(CC) $(CXX_FLAGS) -o myapp $(OBJS) -lcudart

test_matrix: test_matrix.o matrix.o
	$(CC) $(CXX_FLAGS) -o test_matrix test_matrix.o matrix.o -lcudart

test_vector: test_vector.o vector.o
	$(CC) $(CXX_FLAGS) -o test_vector test_vector.o vector.o -lcudart

test_matrix.o: test_matrix.cpp matrix.h
	$(CC) $(CXX_FLAGS) -c test_matrix.cpp

test_vector.o: test_vector.cpp vector.h
	$(CC) $(CXX_FLAGS) -c test_vector.cpp

matrix.o: matrix.cu matrix.h cuda_helpers.h
	$(NVCC) $(CUDA_FLAGS) -c matrix.cu

vector.o: vector.cu vector.h cuda_helpers.h
	$(NVCC) $(CUDA_FLAGS) -c vector.cu

clean:
	rm -f *.o myapp test_matrix test_vector
