/*
 * network.c - 网络通信实现（使用TCP socket）
 */

#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

// 初始化网络连接
bool network_init(NetworkConnection* conn, NetRole role, const char* ip, int port) {
    memset(conn, 0, sizeof(NetworkConnection));
    conn->role = role;
    conn->port = port;
    conn->connected = false;
    conn->socket_fd = -1;
    conn->peer_socket_fd = -1;

    if (ip) {
        strncpy(conn->peer_ip, ip, sizeof(conn->peer_ip) - 1);
    }

    // 创建socket
    conn->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (conn->socket_fd < 0) {
        perror("创建socket失败");
        return false;
    }

    // 设置socket选项
    int opt = 1;
    if (setsockopt(conn->socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("设置socket选项失败");
        close(conn->socket_fd);
        return false;
    }

    return true;
}

// 连接到服务器（客户端）
bool network_connect(NetworkConnection* conn) {
    if (conn->role != NET_ROLE_CLIENT) {
        fprintf(stderr, "只有客户端可以调用network_connect\n");
        return false;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(conn->port);

    if (inet_pton(AF_INET, conn->peer_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "无效的IP地址: %s\n", conn->peer_ip);
        return false;
    }

    printf("正在连接到 %s:%d...\n", conn->peer_ip, conn->port);

    if (connect(conn->socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("连接失败");
        return false;
    }

    conn->connected = true;
    printf("成功连接到服务器！\n");

    // 发送连接确认包
    NetworkPacket packet;
    packet.type = PACKET_CONNECT;
    send(conn->socket_fd, &packet, sizeof(packet), 0);

    return true;
}

// 监听连接（服务器）
bool network_listen(NetworkConnection* conn) {
    if (conn->role != NET_ROLE_SERVER) {
        fprintf(stderr, "只有服务器可以调用network_listen\n");
        return false;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(conn->port);

    // 绑定
    if (bind(conn->socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("绑定失败");
        return false;
    }

    // 监听
    if (listen(conn->socket_fd, 1) < 0) {
        perror("监听失败");
        return false;
    }

    printf("等待玩家2连接 (端口 %d)...\n", conn->port);

    // 接受连接
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    conn->peer_socket_fd = accept(conn->socket_fd, (struct sockaddr*)&client_addr, &client_len);

    if (conn->peer_socket_fd < 0) {
        perror("接受连接失败");
        return false;
    }

    // 获取客户端IP
    inet_ntop(AF_INET, &client_addr.sin_addr, conn->peer_ip, sizeof(conn->peer_ip));
    printf("玩家2已连接: %s\n", conn->peer_ip);

    conn->connected = true;

    // 接收连接确认包
    NetworkPacket packet;
    recv(conn->peer_socket_fd, &packet, sizeof(packet), 0);

    return true;
}

// 发送动作
bool network_send_action(NetworkConnection* conn, TankAction action) {
    if (!conn->connected) return false;

    NetworkPacket packet;
    packet.type = PACKET_ACTION;
    packet.action = action;

    int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;
    ssize_t sent = send(sock, &packet, sizeof(packet), MSG_NOSIGNAL);

    if (sent < 0) {
        perror("发送数据失败");
        conn->connected = false;
        return false;
    }

    return true;
}

// 接收动作
bool network_recv_action(NetworkConnection* conn, TankAction* action) {
    if (!conn->connected) return false;

    NetworkPacket packet;
    int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;

    // 设置非阻塞模式
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    ssize_t received = recv(sock, &packet, sizeof(packet), 0);

    // 恢复阻塞模式
    fcntl(sock, F_SETFL, flags);

    if (received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 没有数据可读
            return false;
        }
        perror("接收数据失败");
        conn->connected = false;
        return false;
    } else if (received == 0) {
        // 连接关闭
        printf("对方已断开连接\n");
        conn->connected = false;
        return false;
    }

    if (packet.type == PACKET_ACTION) {
        *action = (TankAction)packet.action;
        return true;
    }

    return false;
}

// 关闭连接
void network_close(NetworkConnection* conn) {
    if (conn->connected) {
        NetworkPacket packet;
        packet.type = PACKET_DISCONNECT;

        int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;
        send(sock, &packet, sizeof(packet), 0);
    }

    if (conn->peer_socket_fd >= 0) {
        close(conn->peer_socket_fd);
    }
    if (conn->socket_fd >= 0) {
        close(conn->socket_fd);
    }

    conn->connected = false;
}
