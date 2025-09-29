CXX=g++
CXXFLAGS=-std=c++17 -pthread -I./utils -I./mpc -I./ -I/usr/include/eigen3
LDFLAGS=-pthread

.PHONY: all clean test-fc test-primitives

all: test/dealer test/mpc/test_mpc_primitives

test-fc: test/dealer test/nn/test_fc
	@mkdir -p randomness/P0 randomness/P1
	@./test/dealer
	@sleep 1
	@./test/nn/test_fc 0 & ./test/nn/test_fc 1
	@wait
	
test-primitives: test/dealer test/mpc/test_mpc_primitives
	@mkdir -p randomness/P0 randomness/P1
	@./test/dealer
	@sleep 1
	@./test/mpc/test_mpc_primitives 0 & ./test/mpc/test_mpc_primitives 1
	@wait

test/mpc/test_mpc_primitives: test/mpc/test_mpc_primitives.o mpc/mpc.o utils/comm.o
	$(CXX) $(LDFLAGS) -o $@ $^

mpc/mpc.o: mpc/mpc.cpp mpc/mpc.h
	$(CXX) $(CXXFLAGS) -c mpc/mpc.cpp -o mpc/mpc.o

utils/comm.o: utils/comm.cpp utils/comm.h
	$(CXX) $(CXXFLAGS) -c utils/comm.cpp -o utils/comm.o

test/dealer: test/dealer.o
	$(CXX) $(LDFLAGS) -o $@ $^
	
test/dealer.o: test/dealer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f test/dealer test/utils/test_comm test/mpc/test_mpc_primitives test/nn/test_fc utils/*.o mpc/*.o test/mpc/*.o test/nn/*.o test/*.o
