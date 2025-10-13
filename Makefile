CXX=g++
CXXFLAGS=-std=c++17 -pthread -I./utils -I./mpc -I./ -I/usr/include/eigen3
LDFLAGS=-pthread

.PHONY: all clean test-fc test-primitives test-fc-new test-secure-matmul

all: test/dealer test/mpc/test_mpc_primitives

test-secure-matmul: test/dealer_matmul test/mpc/test_secure_matmul
	@mkdir -p randomness/P0 randomness/P1
	@./test/dealer_matmul
	@sleep 1
	@./test/mpc/test_secure_matmul 0 & ./test/mpc/test_secure_matmul 1
	@wait

test-fc-new: test/dealer_fc test/nn/test_fc
	@mkdir -p randomness/P0 randomness/P1
	@./test/dealer_fc
	@sleep 1
	@./test/nn/test_fc 0 & ./test/nn/test_fc 1
	@wait

test-primitives: test/dealer test/mpc/test_mpc_primitives
	@mkdir -p randomness/P0 randomness/P1
	@./test/dealer
	@sleep 1
	@./test/mpc/test_mpc_primitives 0 & ./test/mpc/test_mpc_primitives 1
	@wait

test/nn/test_fc: test/nn/test_fc.o mpc/mpc.o utils/comm.o
	$(CXX) $(LDFLAGS) -o $@ $^
	
test/dealer_fc: test/dealer_fc.o mpc/mpc.o
	$(CXX) $(LDFLAGS) -o $@ $^

test/dealer_matmul: test/dealer_matmul.o mpc/mpc.o utils/comm.o
	$(CXX) $(LDFLAGS) -o $@ $^

test/nn/test_fc.o: test/nn/test_fc.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test/dealer_fc.o: test/dealer_fc.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test/dealer_matmul.o: test/dealer_matmul.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test/mpc/test_mpc_primitives: test/mpc/test_mpc_primitives.o mpc/mpc.o utils/comm.o
	$(CXX) $(LDFLAGS) -o $@ $^

test/mpc/test_secure_matmul: test/mpc/test_secure_matmul.o mpc/mpc.o utils/comm.o
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
	rm -f test/dealer test/dealer_fc test/dealer_matmul test/utils/test_comm test/mpc/test_mpc_primitives test/mpc/test_secure_matmul test/nn/test_fc utils/*.o mpc/*.o test/mpc/*.o test/nn/*.o test/*.o
