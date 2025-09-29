#include "comm.h"
#include <iostream>
#include <thread>
#include <vector>
#include <cstring>
#include "compress.h"

void print_data(const std::vector<uint64_t>& data, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i == data.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

void server_thread() {
    TCPPeer server;
    server.connect("127.0.0.1", 8080, true);
    
    // Test with bitwidth = 3, num_elements = 10
    const int bitwidth = 5;
    const size_t num_elements = 10;
    size_t packed_size = (num_elements * bitwidth + 7) / 8;

    std::vector<uint8_t> packed_data(packed_size);
    server.recv_data(packed_data.data(), packed_size);

    auto unpacked_data = unpack_data(packed_data, num_elements, bitwidth);
    const uint64_t mask = (1ULL << bitwidth) - 1;

    print_data(unpacked_data, "Server received");

    for (size_t i = 0; i < num_elements; ++i) {
        if (unpacked_data[i] != (i & mask)) {
            std::cerr << "Server received incorrect data! " << unpacked_data[i] << " != " << (i & mask) << std::endl;
            return;
        }
    }
    std::cout << "Server received correct data!" << std::endl;

    std::vector<uint64_t> data_to_send(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        data_to_send[i] = (i * 2) % (1 << bitwidth);
    }

    print_data(data_to_send, "Server sending");

    auto packed_to_send = pack_data(data_to_send.data(), num_elements, bitwidth);
    server.send_data(packed_to_send.data(), packed_to_send.size());
    std::cout << "Server sent data!" << std::endl;
}

void client_thread() {
    TCPPeer client;
    // Sleep to give the server time to start
    std::this_thread::sleep_for(std::chrono::seconds(1));
    client.connect("127.0.0.1", 8080, false);

    const int bitwidth = 5;
    const size_t num_elements = 10;
    
    std::vector<uint64_t> data_to_send(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        data_to_send[i] = i;
    }

    print_data(data_to_send, "Client sending");

    auto packed_to_send = pack_data(data_to_send.data(), num_elements, bitwidth);
    client.send_data(packed_to_send.data(), packed_to_send.size());
    std::cout << "Client sent data!" << std::endl;

    size_t packed_size = (num_elements * bitwidth + 7) / 8;
    std::vector<uint8_t> received_packed(packed_size);
    client.recv_data(received_packed.data(), received_packed.size());

    auto unpacked_data = unpack_data(received_packed, num_elements, bitwidth);

    print_data(unpacked_data, "Client received");

    for (size_t i = 0; i < num_elements; ++i) {
        if (unpacked_data[i] != (i * 2) % (1 << bitwidth)) {
            std::cerr << "Client received incorrect data!" << std::endl;
            return;
        }
    }
    std::cout << "Client received correct data!" << std::endl;
}

int main() {
    std::thread t1(server_thread);
    std::thread t2(client_thread);

    t1.join();
    t2.join();

    return 0;
}
