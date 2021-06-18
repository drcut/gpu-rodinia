include ../../common/make.config

SRC = bfs.cpp

EXE = bfs.out


bfs.out: $(SRC)
	clang++ $(SRC) kernels/wrapper_kernel.o kernels/wrapper_kernel2.o \
		library.cpp -o $(EXE) -I../util  \
		-pthread -DTIMING


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
