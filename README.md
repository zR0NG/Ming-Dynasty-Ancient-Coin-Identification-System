# 明代古钱币鉴定系统

基于深度学习的明代古钱币识别 Web 应用，采用 **ResNet50 分类模型 + YOLOv8-seg 实例分割模型** 组合方案，支持单枚钱币正反面识别与多枚钱币同图批量识别，并提供钱币背景知识查询。

> 课程项目：数字图像处理与机器视觉 · 第 6 组（刘振荣、张曜蕾、张舒承）

## 功能特性

| 功能 | 说明 |
| --- | --- |
| 单枚识别 | 上传钱币正面/背面图像，输出钱币类别与置信度 |
| 多枚识别 | 上传包含多枚钱币的图像，由 YOLOv8-seg 分割定位并逐枚裁剪后分类 |
| 知识查询 | 依据 `coin_info.json` 返回对应钱币的朝代、年代、形制等背景信息 |
| 记录浏览 | 按类别归档识别结果，支持历史图片回看 |

## 技术方案

1. **目标检测与分割**：`ultralytics` YOLOv8-seg 对输入图像做实例分割，得到每枚钱币的掩膜与包围框，按框裁剪为单枚子图。
2. **细粒度分类**：`torchvision` ResNet50（`fc` 改为 26 类），输入统一 `Resize(224, 224)` + ImageNet 均值方差归一化。
3. **Web 服务**：Flask 提供页面渲染与文件上传接口，上传图存于 `static/upload/`，归档结果存于 `static/classified/`。

## 项目结构

```
.
├── app.py                  # Flask 应用入口：模型加载、路由、推理逻辑
├── train.py                # ResNet50 分类模型训练脚本
├── coin_info.json          # 26 类钱币的背景知识库
├── training_curves.png     # 训练过程损失/准确率曲线
├── requirements.txt        # Python 依赖
├── templates/              # Jinja2 页面模板
│   ├── base.html
│   ├── index.html
│   ├── single_coin.html
│   ├── multi_coin.html
│   └── classified.html
├── static/                 # 静态资源（运行时生成 upload/ 与 classified/）
├── model/                  # 模型权重目录（权重不纳入版本管理）
└── dataset/                # 训练数据（不纳入版本管理）
    ├── train/
    └── val/
```

## 环境准备

```bash
# 建议 Python 3.9+
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

如需 GPU 推理，请先按本机 CUDA 版本安装对应 `torch` 与 `torchvision`，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 模型权重（需单独获取）

因体积超出仓库承载范围，权重文件未纳入版本管理。运行前请将下列文件放入 `model/` 目录：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `model/best_ming_coins_model.pth` | 约 271 MB | ResNet50 26 类分类权重（必需） |
| `model/yolov8s-seg.pt` | 约 23 MB | YOLOv8-seg 实例分割权重（多枚识别必需） |

> 仓库中已包含 `model/yolov8s-seg.pt`。`best_ming_coins_model.pth` 请通过 Release 附件或网盘获取。
> 缺失权重时 `app.py` 启动即抛出 `FileNotFoundError`，属预期行为。

## 运行

```bash
python app.py
```

默认监听 Flask 开发服务器地址，浏览器访问终端输出的 URL 即可。单次上传大小上限为 32 MB（`app.config['MAX_CONTENT_LENGTH']`）。

## 训练

`train.py` 为 ResNet50 分类模型的训练脚本，主要超参数位于文件头部：

```python
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 26
USE_PRETRAINED = True
```

运行前需修改 `DATA_DIR` 与 `SAVE_DIR` 为本机实际路径：

```bash
python train.py
```

数据集采用 `torchvision.datasets.ImageFolder` 目录约定：

```
dataset/
├── train/
│   ├── chongzhen/
│   ├── daming/
│   └── ...      # 共 26 个类别目录
└── val/
    ├── chongzhen/
    └── ...
```

## 说明

- 本项目为课程作业，模型精度与工程完备度以满足教学演示为目标，未针对生产环境做并发与安全加固。
- `app.py` 中 Flask 以开发模式运行，如需部署请改用 WSGI 服务器（如 gunicorn / waitress）。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。
