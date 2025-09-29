#include "comm.h"
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

TCPPeer::TCPPeer() : sock(-1), client_sock(-1), bytes_sent(0), bytes_received(0) {}

TCPPeer::~TCPPeer() {
    if (sock != -1) {
        close(sock);
    }
    if (client_sock != -1) {
        close(client_sock);
    }
}

bool TCPPeer::connect(const std::string& addr, int port, bool is_server) {
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (is_server) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            throw std::runtime_error("Socket creation failed");
        }

        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            throw std::runtime_error("setsockopt(SO_REUSEADDR) failed");
        }
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
            throw std::runtime_error("setsockopt(SO_REUSEPORT) failed");
        }

        server_addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Bind failed");
        }
        listen(sock, 3);
        socklen_t c = sizeof(struct sockaddr_in);
        struct sockaddr_in client;
        client_sock = accept(sock, (struct sockaddr*)&client, &c);
        if (client_sock < 0) {
            throw std::runtime_error("Accept failed");
        }
    } else { // is_client
        client_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (client_sock == -1) {
            throw std::runtime_error("Socket creation failed for client");
        }
        if (inet_pton(AF_INET, addr.c_str(), &server_addr.sin_addr) <= 0) {
            throw std::runtime_error("Invalid address/ Address not supported");
        }

        int retries = 5;
        bool connected = false;
        for (int i = 0; i < retries; ++i) {
            if (::connect(client_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                connected = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        if (!connected) {
            throw std::runtime_error("Connection Failed after multiple retries");
        }
        // For clients, 'sock' remains -1, preventing double close.
    }
    return true;
}

void TCPPeer::send_data(const void* data, size_t size) {
    if (send(client_sock, data, size, 0) != size) {
        throw std::runtime_error("Send failed");
    }
    bytes_sent += size;
}

void TCPPeer::recv_data(void* data, size_t size) {
    if (recv(client_sock, data, size, MSG_WAITALL) != size) {
        throw std::runtime_error("Recv failed");
    }
    bytes_received += size;
}
