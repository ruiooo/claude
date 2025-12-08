/*
 * network.h - 网络通信模块（用于多人联机对战）
 *
 * 模块说明：
 *   提供基于TCP Socket的多人对战网络通信功能
 *   实现客户端-服务器架构，支持双人实时对战
 *   传输玩家动作和游戏状态同步数据
 *
 * 使用场景：
 *   - 双人联机对战：一个玩家作为服务器，另一个玩家连接
 *   - 本地网络对战：通过局域网IP连接
 *   - 远程对战：通过互联网IP连接（需端口映射）
 *
 * 架构：
 *   - 服务器（玩家1）：监听端口，接受连接，同步游戏状态
 *   - 客户端（玩家2）：连接服务器，发送动作，接收状态
 *   - 数据传输：序列化数据包，TCP可靠传输
 *
 * 协议：
 *   - 连接握手：PACKET_CONNECT
 *   - 动作传输：PACKET_ACTION
 *   - 状态同步：PACKET_SYNC_STATE
 *   - 断开连接：PACKET_DISCONNECT
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>
#include "tank.h"

/*
 * 网络协议常量
 */
#define MAX_PACKET_SIZE 256     // 最大数据包大小（字节）
#define DEFAULT_PORT 12345      // 默认监听端口

/*
 * 网络角色枚举
 * 区分服务器和客户端
 */
typedef enum {
    NET_ROLE_SERVER,  // 服务器（玩家1） - 监听连接，主导游戏状态
    NET_ROLE_CLIENT   // 客户端（玩家2） - 连接服务器，接收状态同步
} NetRole;

/*
 * 数据包类型枚举
 * 定义不同类型的网络消息
 */
typedef enum {
    PACKET_ACTION,      // 动作包：传输玩家的TankAction（移动/射击）
    PACKET_SYNC_STATE,  // 状态同步包：同步坦克位置、血量等状态
    PACKET_CONNECT,     // 连接包：初始握手，建立连接
    PACKET_DISCONNECT   // 断开连接包：正常退出或异常断开
} PacketType;

/*
 * 网络数据包结构体
 * 封装网络传输的所有数据
 */
typedef struct {
    PacketType type;    // 数据包类型，决定如何解析后续数据
    int action;         // 坦克动作（PACKET_ACTION时有效）
    float x, y;         // 坦克位置（PACKET_SYNC_STATE时有效）
    int health;         // 坦克血量（PACKET_SYNC_STATE时有效）
    bool alive;         // 坦克存活状态（PACKET_SYNC_STATE时有效）
} NetworkPacket;

/*
 * 网络连接结构体
 * 管理网络连接的状态和Socket资源
 */
typedef struct {
    int socket_fd;          // 本地Socket文件描述符
    int peer_socket_fd;     // 对端Socket文件描述符（仅服务器使用，客户端为-1）
    NetRole role;           // 网络角色（服务器或客户端）
    bool connected;         // 连接状态标志，true=已连接
    char peer_ip[64];       // 对端IP地址字符串（客户端存储服务器IP）
    int port;               // 端口号（服务器监听端口或客户端连接端口）
} NetworkConnection;

/*
 * 网络功能函数声明
 */

/*
 * 初始化网络连接
 * @param conn: 网络连接结构体指针
 * @param role: 网络角色（服务器或客户端）
 * @param ip: IP地址（客户端：服务器IP；服务器：NULL或"0.0.0.0"监听所有接口）
 * @param port: 端口号（服务器：监听端口；客户端：连接端口）
 * @return: true=成功，false=失败
 * 功能：
 *   - 创建TCP Socket
 *   - 设置Socket选项（如SO_REUSEADDR）
 *   - 保存连接参数
 * 注意：初始化后需调用network_listen（服务器）或network_connect（客户端）
 */
bool network_init(NetworkConnection* conn, NetRole role, const char* ip, int port);

/*
 * 客户端连接到服务器
 * @param conn: 网络连接结构体指针（角色必须是NET_ROLE_CLIENT）
 * @return: true=连接成功，false=连接失败
 * 功能：
 *   - 使用conn->peer_ip和conn->port连接服务器
 *   - 发送PACKET_CONNECT握手包
 *   - 设置connected=true
 * 阻塞行为：阻塞直到连接成功或失败
 */
bool network_connect(NetworkConnection* conn);

/*
 * 服务器监听并接受连接
 * @param conn: 网络连接结构体指针（角色必须是NET_ROLE_SERVER）
 * @return: true=接受连接成功，false=失败
 * 功能：
 *   - 绑定端口并监听
 *   - 等待客户端连接
 *   - 接受连接后保存peer_socket_fd
 *   - 接收PACKET_CONNECT握手包
 *   - 设置connected=true
 * 阻塞行为：阻塞直到有客户端连接
 */
bool network_listen(NetworkConnection* conn);

/*
 * 发送动作数据包
 * @param conn: 网络连接结构体指针
 * @param action: 坦克动作
 * @return: true=发送成功，false=发送失败或连接断开
 * 功能：
 *   - 构造PACKET_ACTION数据包
 *   - 通过Socket发送
 * 用途：玩家操作时发送动作到对方
 */
bool network_send_action(NetworkConnection* conn, TankAction action);

/*
 * 接收动作数据包
 * @param conn: 网络连接结构体指针
 * @param action: 输出接收到的动作
 * @return: true=接收成功，false=接收失败或连接断开
 * 功能：
 *   - 从Socket接收数据包
 *   - 解析PACKET_ACTION数据包
 *   - 输出动作到action参数
 * 阻塞行为：阻塞直到收到数据或超时
 */
bool network_recv_action(NetworkConnection* conn, TankAction* action);

/*
 * 关闭网络连接
 * @param conn: 网络连接结构体指针
 * 功能：
 *   - 发送PACKET_DISCONNECT断开包（如果已连接）
 *   - 关闭Socket（socket_fd和peer_socket_fd）
 *   - 设置connected=false
 * 注意：程序退出前必须调用，确保优雅关闭连接
 */
void network_close(NetworkConnection* conn);

#endif // NETWORK_H
