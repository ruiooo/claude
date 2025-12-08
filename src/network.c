/*
 * network.c - 网络通信实现（基于TCP socket）
 *
 * 模块说明：
 * 本模块实现了多人对战模式的网络通信功能，包括：
 * - TCP socket服务端/客户端
 * - 坦克动作同步
 * - 非阻塞接收
 * - 连接管理
 *
 * 架构：
 * - 服务端（玩家1）：监听端口，等待客户端连接
 * - 客户端（玩家2）：连接到服务端IP
 * - 协议：自定义二进制包（NetworkPacket结构体）
 *
 * 通信流程：
 * 1. 服务端调用network_listen()等待连接
 * 2. 客户端调用network_connect()连接到服务端
 * 3. 双方交换PACKET_CONNECT确认包
 * 4. 每帧发送/接收PACKET_ACTION（坦克动作）
 * 5. 断开时发送PACKET_DISCONNECT
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

/**
 * 初始化网络连接
 *
 * 功能：创建socket并设置基本选项
 *
 * 参数：
 *   conn - 网络连接结构体指针
 *   role - 角色（NET_ROLE_SERVER或NET_ROLE_CLIENT）
 *   ip   - 对端IP地址（客户端必填，服务端可选）
 *   port - 端口号
 *
 * 返回值：
 *   true  - 初始化成功
 *   false - 初始化失败
 *
 * 实现细节：
 *   - 使用TCP协议（SOCK_STREAM）
 *   - 设置SO_REUSEADDR（允许端口重用，避免TIME_WAIT）
 *   - IPv4地址族（AF_INET）
 */
bool network_init(NetworkConnection* conn, NetRole role, const char* ip, int port) {
    // 清零结构体（避免脏数据）
    memset(conn, 0, sizeof(NetworkConnection));

    // 初始化连接参数
    conn->role = role;                    // 服务端或客户端
    conn->port = port;                    // 端口号（默认8888）
    conn->connected = false;              // 初始状态：未连接
    conn->socket_fd = -1;                 // 主socket文件描述符
    conn->peer_socket_fd = -1;            // 对端socket（仅服务端使用）

    // 保存对端IP（客户端需要，服务端可选）
    if (ip) {
        strncpy(conn->peer_ip, ip, sizeof(conn->peer_ip) - 1);
    }

    // ========== 创建socket ==========
    // AF_INET：IPv4地址族
    // SOCK_STREAM：TCP流式socket（可靠、有序、面向连接）
    // 0：自动选择协议（TCP）
    conn->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (conn->socket_fd < 0) {
        perror("创建socket失败");
        return false;
    }

    // ========== 设置socket选项 ==========
    // SO_REUSEADDR：允许地址重用
    // 作用：避免"Address already in use"错误
    // 场景：程序崩溃重启后，旧连接处于TIME_WAIT状态
    int opt = 1;
    if (setsockopt(conn->socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("设置socket选项失败");
        close(conn->socket_fd);  // 清理已创建的socket
        return false;
    }

    return true;
}

/**
 * 连接到服务器（客户端）
 *
 * 功能：作为客户端连接到服务端
 *
 * 参数：
 *   conn - 网络连接结构体指针
 *
 * 返回值：
 *   true  - 连接成功
 *   false - 连接失败
 *
 * 流程：
 *   1. 验证角色（必须是客户端）
 *   2. 构造服务端地址（IP+端口）
 *   3. 调用connect()连接
 *   4. 发送PACKET_CONNECT确认包
 *
 * 阻塞行为：
 *   - connect()会阻塞直到连接成功或超时
 */
bool network_connect(NetworkConnection* conn) {
    // 角色检查：只有客户端可以调用此函数
    if (conn->role != NET_ROLE_CLIENT) {
        fprintf(stderr, "只有客户端可以调用network_connect\n");
        return false;
    }

    // ========== 步骤1：构造服务端地址结构体 ==========
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));  // 清零
    server_addr.sin_family = AF_INET;              // IPv4
    server_addr.sin_port = htons(conn->port);      // 端口号（转换为网络字节序）

    // 转换IP地址：字符串 → 二进制
    // inet_pton: presentation to network
    if (inet_pton(AF_INET, conn->peer_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "无效的IP地址: %s\n", conn->peer_ip);
        return false;
    }

    printf("正在连接到 %s:%d...\n", conn->peer_ip, conn->port);

    // ========== 步骤2：连接到服务端 ==========
    // connect()会阻塞，直到：
    // - 连接成功（返回0）
    // - 连接超时（返回-1，errno=ETIMEDOUT）
    // - 连接被拒绝（返回-1，errno=ECONNREFUSED）
    if (connect(conn->socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("连接失败");
        return false;
    }

    conn->connected = true;
    printf("成功连接到服务器！\n");

    // ========== 步骤3：发送连接确认包 ==========
    // 协议握手：客户端主动发送PACKET_CONNECT
    NetworkPacket packet;
    packet.type = PACKET_CONNECT;
    send(conn->socket_fd, &packet, sizeof(packet), 0);

    return true;
}

/**
 * 监听连接（服务端）
 *
 * 功能：作为服务端监听并接受客户端连接
 *
 * 参数：
 *   conn - 网络连接结构体指针
 *
 * 返回值：
 *   true  - 接受连接成功
 *   false - 监听或接受失败
 *
 * 流程：
 *   1. 验证角色（必须是服务端）
 *   2. 绑定到指定端口
 *   3. 监听（队列长度1，只接受1个客户端）
 *   4. 阻塞等待accept()
 *   5. 接收PACKET_CONNECT确认包
 *
 * 阻塞行为：
 *   - accept()会阻塞直到有客户端连接
 */
bool network_listen(NetworkConnection* conn) {
    // 角色检查：只有服务端可以调用此函数
    if (conn->role != NET_ROLE_SERVER) {
        fprintf(stderr, "只有服务器可以调用network_listen\n");
        return false;
    }

    // ========== 步骤1：构造服务端地址结构体 ==========
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;                  // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY;          // 监听所有网卡接口（0.0.0.0）
    server_addr.sin_port = htons(conn->port);          // 端口号（网络字节序）

    // ========== 步骤2：绑定到端口 ==========
    // bind()：将socket绑定到指定地址和端口
    // 失败原因：端口已被占用、权限不足（<1024端口需要root）
    if (bind(conn->socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("绑定失败");
        return false;
    }

    // ========== 步骤3：开始监听 ==========
    // listen()：将socket标记为被动socket（接受连接）
    // 参数1：队列长度（同时等待连接的最大数量）
    if (listen(conn->socket_fd, 1) < 0) {
        perror("监听失败");
        return false;
    }

    printf("等待玩家2连接 (端口 %d)...\n", conn->port);

    // ========== 步骤4：接受客户端连接 ==========
    // accept()：阻塞等待客户端连接
    // 返回：新的socket文件描述符（用于与客户端通信）
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    conn->peer_socket_fd = accept(conn->socket_fd, (struct sockaddr*)&client_addr, &client_len);

    if (conn->peer_socket_fd < 0) {
        perror("接受连接失败");
        return false;
    }

    // 获取客户端IP地址（用于显示）
    // inet_ntop: network to presentation
    inet_ntop(AF_INET, &client_addr.sin_addr, conn->peer_ip, sizeof(conn->peer_ip));
    printf("玩家2已连接: %s\n", conn->peer_ip);

    conn->connected = true;

    // ========== 步骤5：接收连接确认包 ==========
    // 协议握手：服务端被动接收PACKET_CONNECT
    NetworkPacket packet;
    recv(conn->peer_socket_fd, &packet, sizeof(packet), 0);

    return true;
}

/**
 * 发送动作
 *
 * 功能：发送坦克动作到对端
 *
 * 参数：
 *   conn   - 网络连接结构体指针
 *   action - 要发送的坦克动作（ACTION_IDLE, ACTION_MOVE_UP等）
 *
 * 返回值：
 *   true  - 发送成功
 *   false - 发送失败或未连接
 *
 * 实现细节：
 *   - 封装为PACKET_ACTION包
 *   - 服务端使用peer_socket_fd，客户端使用socket_fd
 *   - 使用MSG_NOSIGNAL避免SIGPIPE信号（对端断开时）
 */
bool network_send_action(NetworkConnection* conn, TankAction action) {
    if (!conn->connected) return false;  // 未连接则跳过

    // 构造动作包
    NetworkPacket packet;
    packet.type = PACKET_ACTION;    // 包类型：动作
    packet.action = action;         // 动作值（0-8）

    // 选择正确的socket描述符
    // 服务端：使用peer_socket_fd（与客户端通信的socket）
    // 客户端：使用socket_fd（连接到服务端的socket）
    int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;

    // 发送数据
    // MSG_NOSIGNAL：防止对端断开时产生SIGPIPE信号（会导致程序崩溃）
    ssize_t sent = send(sock, &packet, sizeof(packet), MSG_NOSIGNAL);

    if (sent < 0) {
        perror("发送数据失败");
        conn->connected = false;  // 标记连接已断开
        return false;
    }

    return true;
}

/**
 * 接收动作
 *
 * 功能：非阻塞接收对端的坦克动作
 *
 * 参数：
 *   conn   - 网络连接结构体指针
 *   action - 输出参数：接收到的坦克动作
 *
 * 返回值：
 *   true  - 成功接收到动作
 *   false - 无数据、连接断开或错误
 *
 * 非阻塞实现：
 *   - 临时设置socket为非阻塞模式（O_NONBLOCK）
 *   - 接收后恢复原模式
 *   - 无数据时立即返回false（避免阻塞游戏循环）
 *
 * 错误处理：
 *   - EAGAIN/EWOULDBLOCK：无数据可读（正常）
 *   - received == 0：对端关闭连接
 *   - received < 0：其他错误
 */
bool network_recv_action(NetworkConnection* conn, TankAction* action) {
    if (!conn->connected) return false;  // 未连接则跳过

    NetworkPacket packet;

    // 选择正确的socket描述符
    int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;

    // ========== 步骤1：保存并设置非阻塞模式 ==========
    // fcntl：文件控制（file control）
    // F_GETFL：获取文件状态标志
    int flags = fcntl(sock, F_GETFL, 0);
    // F_SETFL：设置文件状态标志
    // O_NONBLOCK：非阻塞I/O
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    // ========== 步骤2：非阻塞接收 ==========
    ssize_t received = recv(sock, &packet, sizeof(packet), 0);

    // ========== 步骤3：恢复原阻塞模式 ==========
    fcntl(sock, F_SETFL, flags);

    // ========== 步骤4：错误处理 ==========
    if (received < 0) {
        // 情况1：无数据可读（非阻塞正常情况）
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;  // 不是错误，只是暂无数据
        }
        // 情况2：真实错误（网络异常等）
        perror("接收数据失败");
        conn->connected = false;
        return false;
    } else if (received == 0) {
        // 情况3：对端优雅关闭连接（FIN包）
        printf("对方已断开连接\n");
        conn->connected = false;
        return false;
    }

    // ========== 步骤5：解析动作包 ==========
    if (packet.type == PACKET_ACTION) {
        *action = (TankAction)packet.action;  // 提取动作值
        return true;
    }

    return false;  // 收到非动作包（忽略）
}

/**
 * 关闭连接
 *
 * 功能：优雅关闭网络连接
 *
 * 参数：
 *   conn - 网络连接结构体指针
 *
 * 流程：
 *   1. 发送PACKET_DISCONNECT通知对端
 *   2. 关闭所有socket文件描述符
 *   3. 标记连接为未连接状态
 *
 * 优雅关闭：
 *   - 通知对端（而不是直接close）
 *   - 对端收到后可以清理资源
 */
void network_close(NetworkConnection* conn) {
    // ========== 步骤1：发送断开通知 ==========
    if (conn->connected) {
        NetworkPacket packet;
        packet.type = PACKET_DISCONNECT;  // 断开连接包

        // 选择正确的socket
        int sock = (conn->role == NET_ROLE_SERVER) ? conn->peer_socket_fd : conn->socket_fd;
        send(sock, &packet, sizeof(packet), 0);
    }

    // ========== 步骤2：关闭socket文件描述符 ==========
    // 服务端：需要关闭两个socket（监听socket和通信socket）
    if (conn->peer_socket_fd >= 0) {
        close(conn->peer_socket_fd);  // 关闭通信socket
    }
    // 客户端和服务端：都需要关闭主socket
    if (conn->socket_fd >= 0) {
        close(conn->socket_fd);       // 关闭主socket
    }

    // ========== 步骤3：标记为未连接 ==========
    conn->connected = false;
}
