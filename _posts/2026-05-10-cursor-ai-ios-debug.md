---
title: '让 Cursor AI 在 iOS真机联调环境中自主调试修复 Bug'
date: 2026-05-10 12:00:00 +0800
published: true
permalink: /posts/2026/05/cursor-ai-ios-debug/
tags:
  - Cursor
  - iOS
  - 调试
  - AI
---

本文记录一次真实客户端问题排查：**现象是 App 的【任务列表】界面里多条「等待中 / 正在识别」时，网络层出现高频轮询**；手动排查思路是在网络请求下断点，逐步调试找到错误原因。但是，随着 AI 能力的提升，让 AI 在真机环境中自动发现问题根因，和给出解决方案，是一个很高效、自主进化迭代的工作流。本文目标不是「让 AI 猜一猜」问题原因，而是把 **运行时可观测数据** 接进工作流，让 Cursor 里的模型在 **有证据** 的前提下走完 **加埋点 → 根因定位 → 代码修复 → 编译验证 → 删埋点** 的闭环。

## 一、问题

任务列表中同时存在多条 **「等待中」**、**「正在识别」** 等活跃状态；XCode 调试面板表明网络请求高频轮询，循环没有停止。

![image-20260510120900549](/images/md-img/image-20260510120900549.png)

## 二、Cursor 调试真机接入

步骤流程：创建 skill、调用 skill、写好提示词，要求 AI 找出根本原因和修复方案，开发者作为真机运行测试员。

在 cursor 中创建 skill，我用的 skill 文件内容如下，读者可参考大致内容，让 AI 自己写一份 skill，让 Cursor 自己创建新的 skill。

````
---
name: ios-hotspot-http-tracking
description: Guides iOS real-device analytics tracking tests that report to a Python HTTP server running on a hotspot-connected Mac. Use when the user mentions iOS tracking, event reporting, real-device network debugging, hotspot testing, local HTTP server, or requests event capture on Mac.
---

# iOS Hotspot HTTP Tracking

## 目标

在 iOS 真机上报埋点时，把请求发送到运行在 Mac 上的 Python HTTP server（通过手机热点互联），并完成连通性与数据落地验证。

## 必做原则

1. 先向用户索取 **热点 Mac 的 IP 地址**，再继续后续步骤。
2. 默认使用 `http://<mac_ip>:<port>`，除非用户明确要求 HTTPS。
3. 每次修改上报地址后，都执行一次端到端验证（客户端触发事件 + 服务端看到请求）。

## 标准流程

复制并跟踪以下清单：

```markdown
Task Progress:
- [ ] 1. 索取热点 Mac IP
- [ ] 2. 确认监听端口与上报路径
- [ ] 3. 启动 Python HTTP server
- [ ] 4. 配置 iOS 埋点上报地址
- [ ] 5. iPhone 连接热点并触发埋点
- [ ] 6. 校验服务端收到请求并回传结果
```

### 1) 索取热点 Mac IP（必须）

先向用户确认：

- 热点 Mac IP（例如 `172.20.10.2`）
- 监听端口（默认可用 `8000`）
- 上报路径（例如 `/track`）
- 请求方法与载荷格式（GET/POST，JSON/表单）

如用户未提供端口或路径，使用默认：

- 端口：`8000`
- 路径：`/track`

### 2) 启动 Python HTTP server

优先使用可打印请求内容的最小服务。示例：

```python
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = 8000

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("GET", self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        print("POST", self.path)
        print("Headers:", dict(self.headers))
        print("Body:", body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

if __name__ == "__main__":
    print(f"Listening on {HOST}:{PORT}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
```

运行：

```bash
python3 server.py
```

### 3) 配置 iOS 上报地址

将埋点基地址改为：

`http://<mac_ip>:<port>`

并拼接上报路径，例如：

`http://172.20.10.2:8000/track`

### 4) 真机联调验证

1. iPhone 连接该 Mac 开启的热点。
2. 在 iOS App 中触发一条可识别埋点（如 `test_event`）。
3. 观察 Python server 输出，确认路径、方法、请求体正确。
4. 若无请求到达，按排障步骤处理。

## 快速排障

- IP 不通：确认手机和 Mac 在同一热点网络、IP 未变化。
- 端口不通：确认 server 绑定 `0.0.0.0` 且端口一致。
- 路径错：确认客户端路径与服务端预期一致。
- 请求未发：检查 iOS 埋点开关、触发条件、环境开关。
- 解析失败：先打印原始 body，再对齐 JSON 字段。

## 输出要求

完成后统一给用户以下结果：

1. 使用的上报 URL（含 IP、端口、路径）
2. 已验证的事件名与请求示例
3. 服务端接收截图/日志要点（文本摘要即可）
4. 若失败，给出下一步最小排障动作（1-2 条）

````

在对话中调用技能，我用到的提示词如图：

![image-20260510120739525](/images/md-img/image-20260510120739525.png)

注意：左下角的模式改成 Debug

**Cursor 能替开发者省下的体力**在于：**直接改仓库、跑 `xcodebuild`、多文件一致重构**；前提是开发者把 **「证据从真机送回 Mac」** 这条管道接好——Skill + `pyserver` 解决的是「管道怎么搭」。

## 三、Cursor自主治理

cursor read能力与真机调试日志打通后，具备了观察真机运行时信息的能力，得益于具备原生的 Debug 模式，cursor agent 读完日志，分析原因，做代码更改，弹调试对话框指引开发者真机操作，根据反馈继续迭代或终止。

以下是观察到AI的自主治理的一些信息：

![image-20260510123706919](/images/md-img/image-20260510123706919.png)

![image-20260510123649886](/images/md-img/image-20260510123649886.png)

Cursor还给实现了解决方案：

![image-20260510123807293](/images/md-img/image-20260510123807293.png)

修复代码后的真机界面如图：

![image-20260510124255281](/images/md-img/image-20260510124255281.png)

## 四、结语

Cursor 这类工具最适合的角色是 **「在约束下批量改代码、跑构建、做多文件一致性」**；**根因**仍然要靠 **运行时证据** 约束，否则模型只能输出「概率性正确」的补丁。

把 **真机 → 热点 → Mac 上的小日志服务 → JSON / 结构化日志** 这条链路接好，本质上是给 AI 配了一副「慢放眼镜」：**先对齐事实，再对齐代码**。这套流程与具体业务无关，换到 Android、Electron 调试，同样成立。
