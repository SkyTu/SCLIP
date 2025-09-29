#pragma once

#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

class TCPPeer {
public:
    TCPPeer();
    ~TCPPeer();

    bool connect(const std::string& addr, int port, bool is_server);
    void send_data(const void* data, size_t size);
    void recv_data(void* data, size_t size);

    // Communication statistics
    size_t get_bytes_sent() const { return bytes_sent; }
    size_t get_bytes_received() const { return bytes_received; }
    void reset_stats() { bytes_sent = 0; bytes_received = 0; }

private:
    int sock;
    int client_sock;
    struct sockaddr_in server_addr;
    
    // Statistics counters
    size_t bytes_sent;
    size_t bytes_received;
};

