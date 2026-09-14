 # 太阳能AI助手前端重构说明

本项目基于 Gradio 实现，旨在为太阳能电池相关的AI对话、参数预测、老化预测、钙钛矿参数与带隙预测等功能提供统一、易扩展的前端界面。

## 目录结构与模块说明

```
web/
├── app.py                # 入口文件，只负责组装和启动所有Tab
├── config.py             # 全局配置（如API_BASE_URL）
├── utils/
│   ├── __init__.py
│   └── logger.py         # 日志配置
├── api/                  # 各功能Tab的后端API交互与业务逻辑
│   ├── __init__.py
│   ├── chat.py           # AI对话API逻辑
│   ├── solar.py          # 太阳能参数预测API逻辑
│   ├── aging.py          # 老化预测API逻辑
│   ├── perovskite.py     # 钙钛矿参数预测API逻辑
│   └── bandgap.py        # 钙钛矿带隙预测API逻辑
├── ui/                   # 各功能Tab的UI与事件绑定
│   ├── __init__.py
│   ├── chat.py           # AI对话Tab UI
│   ├── solar.py          # 太阳能参数预测Tab UI
│   ├── aging.py          # 老化预测Tab UI
│   ├── perovskite.py     # 钙钛矿参数预测Tab UI
│   └── bandgap.py        # 钙钛矿带隙预测Tab UI
└── readme.md             # 项目说明文档
```

## 主要改动说明

1. **入口文件精简**：
   - `app.py` 只负责导入和组装各Tab，无任何业务逻辑。
   - 各Tab通过 `build_xxx_tab()` 方式注册，结构清晰。

2. **UI与业务逻辑彻底分离**：
   - `ui/` 目录下每个Tab一个文件，负责UI控件、事件绑定。
   - `api/` 目录下每个Tab一个文件，负责与后端API交互、数据处理、参数打包、图片解码等。

3. **全局配置与工具独立**：
   - `config.py` 统一管理API地址等全局配置。
   - `utils/logger.py` 统一日志配置。

4. **易于扩展和维护**：
   - 每个功能模块独立，便于后续功能扩展、多人协作和单独测试。
   - 新功能只需在 `ui/` 和 `api/` 下增加对应文件，并在 `app.py` 注册即可。

## 如何增加新的功能Tab

1. **在 `api/` 目录下新建API逻辑文件**
   - 例如：`api/newfeature.py`
   - 实现与后端API的交互、参数处理、结果格式化等。

2. **在 `ui/` 目录下新建UI文件**
   - 例如：`ui/newfeature.py`
   - 实现 `build_newfeature_tab()`，负责Tab的UI控件、事件绑定，调用 `api/newfeature.py` 的接口。

3. **在 `app.py` 中注册新Tab**
   - 导入 `from ui.newfeature import build_newfeature_tab`
   - 在 `with gr.Blocks(...) as demo:` 里调用 `build_newfeature_tab()`

4. **（可选）在 `config.py` 增加新配置项**
   - 如有新的API地址、全局参数等，可统一在 `config.py` 管理。

5. **（可选）在 `utils/` 增加通用工具**
   - 如有通用的防抖、格式化、校验等工具，可放在 `utils/` 下。

## 开发建议

- **保持UI与业务逻辑分离**，UI文件只负责界面和事件，所有数据处理、API交互都放在api目录。
- **参数顺序和命名要统一**，UI和API间参数传递要保持一致。
- **充分利用Gradio的事件绑定和组件复用**，减少重复代码。
- **多人协作时可按Tab分工**，互不影响。
- **如需支持多语言或主题切换，可在UI层统一处理。**

## 运行方式

```bash
python app.py
```

## 结语

本重构极大提升了前端的可维护性和可扩展性。后续如需增加新功能、优化界面或对接新API，只需按模块扩展即可，无需大幅改动主流程。
