---
title: AI视频生成/视频工作流教程文档
tags: Technical Manual
---
# 提示词模版、示例：

## 文生图提示词模版：
``` bash
# --- 1. 自然语言描述块 (Visual Script) ---
# 用自然语言描绘画面，重点在于“物理属性”和“光影质感”
Final Prompt (EN): |
  [核心动态瞬间]: [主体的具体动作/位置] + [环境的瞬间状态 (如飞溅的雪/光晕)].
  [环境构建]: [前景细节] + [中景主体] + [远景地貌]; [明确排除的元素 (如无浮冰)].
  Camera: [镜头类型 (如 FPV/Drone)], [焦段感 (如 18-24mm)], [景深 (如 Deep DOF)]; [构图方式 (如 荷兰角/三分法)].
  Lighting: [主光源 (冷/暖)] + [轮廓光/边缘光] + [体积光/雾气] + [特殊光效 (如 钻石尘)].
  Color: [主色调 (如 克制的蓝白)], [高光色调], [对比度风格 (如 Filmic contrast)], [高光滚降 (Smooth rolloff)].
  Texture: [微观材质 (如 霜冻金属)], [环境材质 (如 风蚀雪纹)].
  Mood: [情绪关键词], [渲染风格 (Photorealistic, Cinematic, No heavy HDR)].

# --- 2. 结构化参数块 (Technical Parameters) ---
# 给 AI 的硬性指标，确保画质和风格的一致性
Parameters:
  aspect_ratio: [比例，如 16:9]
  camera: [镜头关键词，如 fpv-cinema, dutch-tilt, low-angle]
  lens_mm: [具体焦段，如 18-24]
  aperture: [光圈数值，如 f/4.5]
  lighting: [光照关键词组合，如 sunrise rim + skylight + diamond dust]
  grade: [调色风格，如 filmic contrast, cold_contrast with warm highlights]
  texture: [关键材质，如 sastrugi ridges, frosted paint]
  style_tags: [风格标签组合]
  negative: [负面提示词：明确禁止的物体 + 画质缺陷 (如 neon HDR, plastic snow)]
```
---

## 图生视频提示词模版：
``` bash
# --- 1. 运动控制 (Motion Control) ---
Parameters:
  duration_s: [时长，通常为 5s 或 8s]
  # 核心：使用箭头 "→" 定义运镜的序列
  movement: [动作A] → [动作B] → [动作C]
  # (例: FPV banked-orbit → low skim → steep climb)
  
  motion_curve: [物理速度曲线] ([备注：哪里加速，哪里减速])
  # (例: ease-in-out (subtle mid-shot speed ramp))
  
  stabilization: [防抖模式] (如 fpv-smooth / cinematic-smooth / handheld-raw)
  parallax: enabled, strength: [数值 0.1-1.0] ([备注：利用什么物体产生视差])

# --- 2. 画面细节控制 (Visual Consistency) ---
  depth_of_field: [景深设置], focus_target: [焦点始终跟随的主体]
  grain: [颗粒感强度] (需与原图一致)
  grade: [调色风格] (需与原图一致)

# --- 3. 分秒导演脚本 (Shot Notes) ---
# 最关键部分：按秒拆解画面内容，防止 AI 动作变形
  shot_notes: "
    0–1s: [起始动作] + [瞬间视觉冲击 (如掠过镜头)];
    1–Xs: [中间过渡动作] + [运镜轨迹描述 (如 80°倾斜盘旋)];
    X–Ys: [高潮动作] + [环境揭示 (如 拉升展现广阔平原)];
    Y–8s: [结尾落幅] + [最终构图保持 (如 驶向太阳)]."
```
---
## 文生视频提示词模版：
``` bash
# --- 1. 动态叙事脚本 (Generative Script) ---
Final Prompt (EN): |
  [事件全貌]: A continuous shot showing [主体] performing [复杂的物理动作序列].
  
  [时空演变]:
  - Start: [起始画面描述] (环境、光影、主体状态).
  - Action: [中间发生的物理变化] (如：雪崩发生、物体破碎、人物从跑变为跳).
  - End: [结束画面描述] (最终的落幅和环境变化).
  
  Camera Movement: [运镜方式] following the action, [镜头特征 (如 handheld shake)].
  Physics/Details: [特殊的物理细节描述] (如：雪花飞溅的轨迹、衣服的褶皱变化).
  Visual Bible: [直接复用文生图的 Lighting, Color, Texture 定义，确保风格不跑偏].

# --- 2. 生成参数 (Generation Params) ---
Parameters:
  aspect_ratio: 16:9
  duration_s: [时长]
  physics_simulation: [物理模拟关键词，如 fluid dynamics, collision, cloth simulation]
  consistency_guard: [风格约束关键词，如 photorealistic, consistent character style]
  negative: [morphing, disappearing objects, extra limbs, logic errors, defying gravity]
```
---
