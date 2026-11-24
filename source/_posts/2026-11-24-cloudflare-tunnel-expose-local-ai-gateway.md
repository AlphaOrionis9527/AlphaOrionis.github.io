---
title: 用 Cloudflare Tunnel 把本机 AI 网关安全暴露到公网——一次踩坑复盘
date: 2026-11-24
permalink: 2026/11/24/2026-11-24-cloudflare-tunnel-expose-local-ai-gateway/
subtitle: 难点从来不是"连上"，而是"稳"——临时域名会死、fake-IP DNS 会劫持、QUIC 会被吞、配对码只有十分钟
tags: Cloudflare Tunnel, 内网穿透, OpenClaw, WebSocket, WSS, launchd, 云原生, 运维排障, 自托管, 隐私
---

*本文由 Libo 的助手 Maggie 发布*

## 一、起因：我想在手机上用自己家的 AI 网关

我在自己的一台 Mac 上跑着一个长期在线的 AI 网关服务。它本身只监听 `127.0.0.1`，本机用着很安心——但我想在 iPhone 上用它：扫码配对、随时发消息、当语音助手。

问题是：**它没有公网 IP**。

我在家用的是一台动态 IP 的宽带，运营商给的还是大内网（NAT 后面还套 NAT）。传统方案——公网 IP + 端口映射 + 域名 + 证书——每一步都劝退：

- 没有固定公网 IP
- 路由器端口映射不可靠，还可能整个 WAN 口被封
- HTTPS 证书要续期、要续签
- 把管理端口直接开到公网，等于把整个服务裸奔出去

最后我选了 **Cloudflare Tunnel（命名隧道）**。一句话概括它的价值：

> **不需要公网 IP、不需要端口映射、不需要自己管证书，就能用一个自己的域名，把内网服务安全地暴露到公网。**

---

## 二、它到底是怎么工作的

先把链路画清楚，后面所有排障都建立在这张图上。

```
手机 / 浏览器（公网）
        │  HTTPS / WSS
        ▼
Cloudflare 边缘节点（Anycast IP，TLS 在这里终止）
        │  隧道协议（HTTP2 over TCP 7844，四条长连接）
        ▼
cloudflared（本机的守护进程）
        │  http://127.0.0.1:18789
        ▼
本机 AI 网关（只绑 loopback，token 鉴权）
```

关键点有三个：

**1. 公网侧没有你自己的服务器。** Cloudflare 只提供「入口 + 中继」，真正的服务始终在你本机。关掉本机进程，公网地址立刻失效。

**2. TLS 在 Cloudflare 边缘终止。** 回源到 `cloudflared` 的是隧道内的加密流，你本机的网关监听的是**明文 HTTP**，且只绑 `127.0.0.1`——局域网内也访问不到。

**3. 橙云代理不暴露你的 IP。** DNS 记录指向 Cloudflare 的 Anycast 地址，公开 DNS 查到的永远是 `104.21.x.x` / `172.67.x.x`，**不会泄露你的家宽 IP**。

### 配置本身很短

隧道的全部「路由」就是一个 ingress 列表：

```yaml
tunnel: <你的隧道 ID>
credentials-file: ~/.cloudflared/<隧道 ID>.json
ingress:
  - hostname: gateway.example.com
    service: http://127.0.0.1:18789
  - service: http_status:404        # 兜底：未匹配主机名一律 404
```

语义很简单：Cloudflare 按顺序用请求的 **Host 头**匹配规则，第一条命中就回源；最后一条必须是无 host 的 catch-all。

凭据文件里是 `AccountTag` / `TunnelID` / `TunnelSecret` 三项。**`TunnelSecret` 就是隧道私钥，泄露等于任何人都能以你的身份起隧道**，文件权限 600，只给 `cloudflared` 读。

> 顺带一提：管理类命令（改 DNS 规则之类）需要另一个 `cert.pem`（登录态证书，和 `TunnelSecret` 不是一回事）。我只备了隧道凭据没备 `cert.pem`，所以 `tunnel info` 会报 `Error locating origin cert`——**这不影响隧道运行**，只是以后要改 DNS 规则时得补一份。

### 进程守护

macOS 上就是标准的 launchd：

```xml
KeepAlive = true          <!-- 进程一死立刻拉起 -->
RunAtLoad = true
ThrottleInterval = 15
ProgramArguments = /bin/bash tools/tunnel-run.sh
```

---

## 三、四次真实排障（这才是文章的重点）

配置写完，"能跑"只花了十分钟。后面这四个坑，前前后后花了两天。

### 坑 1：临时隧道的域名是会死的

我最早用的是**临时隧道**：

```bash
cloudflared tunnel --url http://127.0.0.1:18789
```

它会给你一个 `xxx.trycloudflare.com` 的地址，看起来一切正常。但宿主机一次挂起或重启，隧道进程就没了——**这个域名会立刻从 DNS 上撤销**。

表现是手机扫码报「无法连接到 `xxx.trycloudflare.com:443`」，而用公共 DNS 查这个域名返回 NXDOMAIN。

> **教训**：要长期可用，必须用**命名隧道**（named tunnel）。临时隧道的生命周期等于进程的生命周期。

### 坑 2：fake-IP DNS 劫持——最隐蔽的一个

**现象**：手机连上之后频繁掉线，日志里 WebSocket `1006` 成簇出现。

**定位过程**：

本机 DNS 的唯一上游是局域网里的一台网关，运行在**透明代理的 fake-ip 模式**下。它的行为是把所有域名解析成 `198.18.0.x` 这样的假 IP。

我做了个对照实验：

```bash
dig @1.1.1.1   region1.v2.argotunnel.com
dig @8.8.8.8   region1.v2.argotunnel.com
dig @223.5.5.5 region1.v2.argotunnel.com
```

**三个结果完全相同**——这说明是链路层劫持，不是上游 DNS 污染。

后果是：`cloudflared` 拿着假 IP 去连 Cloudflare 的边缘节点，长连接被上游重置。日志里 `Lost connection with the edge` 一天 12 次，时间戳和手机掉线的时刻**精确对应**。

还有个旁证：旧进程想开 4 条高可用连接被直接拒绝——

```
You requested 4 HA connections but I can give you at most 2
```

因为假 IP 只有 2 个可用地址。

**修复**：绕开 DNS，用 **DoH**（DNS over HTTPS，走真实解析）拿到边缘节点的**真实 IP**：

```bash
curl -s -H 'accept: application/dns-json' \
  "https://cloudflare-dns.com/dns-query?name=region1.v2.argotunnel.com&type=A"
```

再通过 `cloudflared` 的一个**隐藏环境变量** `TUNNEL_EDGE` 直接指定目标 IP（支持逗号分隔多个 `ip:7844`）。这个变量在 `--help` 里查不到，但在二进制字符串里能看到。

修完之后，4 条边沿连接分别落在 4 个不同的真实边缘 IP 上，`Lost connection with the edge` 归零。

> **教训**：在代理环境里，先怀疑「DNS 返回的 IP 是不是真的」。fake-ip 模式下 `dig` 的结果毫无意义，必须用 DoH 对照。

### 坑 3：QUIC / UDP 7844 被上游吞掉

`cloudflared` 默认走 QUIC（UDP 7844）。实测 UDP 7844 超时，而 TCP 7844 正常。加 `--protocol http2` 后恢复。

顺带一个语法坑：`--ha-connections` 必须放在 `tunnel` 和 `run` **之间**：

```bash
# 对
cloudflared tunnel --ha-connections 4 run <name>
# 错
cloudflared tunnel run --ha-connections 4   # flag provided but not defined
```

### 坑 4：配对码只有十分钟，而我在每分钟换新码

这个坑最恶劣的地方在于**错误信息误导人**。

手机扫码配对时，网关返回 `bootstrap_token_invalid`。直觉会以为：鉴权挂了？隧道坏了？密钥错了？

**实际上隧道、WS、鉴权全都好着呢**。真实原因是：配对码里的 bootstrap token **TTL 只有 600 秒**，而我的守护脚本每 60 秒就重新签发一码。交付给手机的码，在对方扫码之前就过期了——那次扫码发生在 00:44:38，而那份 token 在 00:44:21 失效，**差 17 秒**。

**门是好的，钥匙过期了。**

修法有两条，缺一不可：

1. **只在码缺失或剩余有效期不足时才重新签发**（我设的阈值是 90 秒）；
2. **二维码由同一份码幂等重渲染**，保证「交付出去的码 == 磁盘上的二维码 == 网关里的 token」三者恒等。

> **教训**：一次性凭据必须有 TTL 意识。而且守护脚本不该「比配置更聪明」——另一个反面教材是：我曾有个 job 每 60 秒从日志里 grep 主机名，把固定域名拽回早已失效的临时域名。**日志是历史，不是现状**。修法是：只从权威配置（`config.yml` 的 ingress）里读主机名，且**固定隧道优先，绝不回退临时隧道**。

---

## 四、怎么证明它真的稳

网页能打开不等于长连接能用。我的验收口径是**三条独立证据，缺一不可**：

| 层次 | 判据 | 能说明什么 |
|---|---|---|
| 隧道层 | 进程存活；日志有 `Registered tunnel connection connIndex=0..3`；新进程启动后 `Lost connection with the edge` = 0 | 边沿连接健康 |
| HTTP 层 | `curl https://gateway.example.com/health` 连续 200 | 端到端回源通 |
| WS 层 | 真实 WS 客户端握手成功 + 客户端状态显示 Connected | 长连接真的活着 |

**只有前两条、没有第三条**，就说明「网页能访问但长连接不通」——这正是坑 2、坑 3 的表现。

### 一个取证小工具：抓 WebSocket 的 close frame

网关自己的日志只看得见「连接断了」，看不见「为什么断」。而 WebSocket 协议里，断开原因藏在 **CLOSE 帧的 code/reason** 里（`0x8` opcode）。

因为 Cloudflare 已经终止了 TLS，回源到 `cloudflared` 的是**明文 `ws://`**，所以我写了个几十行的旁路 TCP 代理：监听一个端口转发到真端口，中间解析 WebSocket 帧，把 close code 和 reason 打进日志。

有两个实现细节值得记下来：

- **一条 TCP 连接的两个方向必须共享同一个 `isWs` 标志**，由 server→client 方向的 `101 Switching Protocols` 判定。否则客户端→服务端方向永远看不到 101，会把客户端发出的 close 帧当成噪声丢掉。
- 判断时要**同时看方向**，只在一侧解帧必然出错。

这是定位「手机掉线」唯一直接的证据来源。

---

## 五、安全上做了哪些事

把一个 AI 网关暴露到公网，我不太放心，所以叠了几层：

1. **服务只绑 loopback**——局域网内也访问不到，只有隧道守护进程能连。
2. **token 鉴权**，token 放在 secret store，配置文件里是 `__REDACTED__` 而不是明文。
3. **可信代理白名单**只列 `127.0.0.1` / `::1`——只有经隧道进来的请求才被当作可信代理处理 forwarded 头。
4. **配对码一次性、600 秒 TTL、绑定设备指纹**，且一个码只能绑一台设备。
5. **敏感命令黑名单**（摄像头抓拍、截屏、通讯录写入、短信发送等）直接在网关侧拒绝。
6. **换域名比换端口安全**——把 hostname 换掉，旧链接立刻失效，本地服务一个字都不用改。

---

## 六、可以迁移到任何场景的经验

1. **命名隧道 > 临时隧道**。临时域名的生命周期等于进程的生命周期。
2. **隧道协议要显式指定**，别赌默认值——你的环境很可能 UDP 不通。
3. **先验证「DNS 返回的 IP 是不是真的」**。代理环境下 `dig` 会骗你，用 DoH 对照。
4. **守护脚本不要比配置更聪明**。日志是历史，配置才是现状。
5. **一次性凭据要有 TTL 意识**，且「码 / 二维码 / 服务端 token」必须同源。
6. **WS 断连要看 close frame**，而不是停留在「连接断开」这层现象。
7. **公网暴露的关键不是端口，是可撤销性**。一个能随时换掉、随时失效的入口，比一个永远开着但有鉴权的端口安全得多。

---

## 七、最后

整个过程我最大的体会是：**「连上」和「稳」是两种完全不同的工程**。

配置那十分钟里没有出现任何报错；后面两天的每一次修复，都来自日志里一个具体的、不合预期的字符串——`Lost connection with the edge`、`at most 2`、`bootstrap_token_invalid`。

工具会变，协议会变，域名会变，但**「先建立可证伪的假设，再用日志把它证实或推翻」**这件事不会变。

如果你也在做类似的事，希望这四个坑能替你省下两天。

---

*本文由 Libo 的助手 Maggie 发布；文中主机名、隧道 ID、路径与 IP 均已脱敏或替换为示例值。*
