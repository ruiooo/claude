/*
 * network.h - 网络通信模块（用于多人联机对战）
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>
#include "tank.h"

#define MAX_PACKET_SIZE 256
#define DEFAULT_PORT 12345

// 网络角色
typedef enum {
    NET_ROLE_SERVER,  // 服务器（玩家1）
    NET_ROLE_CLIENT   // 客户端（玩家2）
} NetRole;

// 数据包类型
typedef enum {
    PACKET_ACTION,      // 动作包
    PACKET_SYNC_STATE,  // 状态同步包
    PACKET_CONNECT,     // 连接包
    PACKET_DISCONNECT   // 断开连接包
} PacketType;

// 网络数据包
typedef struct {
    PacketType type;
    int action;         // 坦克动作
    float x, y;         // 位置（用于同步）
    int health;         // 血量（用于同步）
    bool alive;         // 是否存活
} NetworkPacket;

// 网络连接
typedef struct {
    int socket_fd;
    int peer_socket_fd;  // 仅服务器使用
    NetRole role;
    bool connected;
    char peer_ip[64];
    int port;
} NetworkConnection;

// 函数声明
bool network_init(NetworkConnection* conn, NetRole role, const char* ip, int port);
bool network_connect(NetworkConnection* conn);
bool network_listen(NetworkConnection* conn);
bool network_send_action(NetworkConnection* conn, TankAction action);
bool network_recv_action(NetworkConnection* conn, TankAction* action);
void network_close(NetworkConnection* conn);

#endif // NETWORK_H
