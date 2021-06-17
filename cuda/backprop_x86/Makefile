include ../../common/make.config

# C compiler
CC = clang
CC_FLAGS = -g  -O2


backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	clang++ $(CC_FLAGS) backprop.o facetrain.o imagenet.o \
			backprop_cuda.o kernels/wrapper_bpnn_forward.o \
			kernels/wrapper_bpnn_adjust.o library.cpp \
			-o backprop -lpthread -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cpp backprop.h
	clang++ $(CC_FLAGS) -c backprop_cuda.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
