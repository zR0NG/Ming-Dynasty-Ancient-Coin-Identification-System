import os
import json
import time
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# ========== 初始化核心配置 ==========
app = Flask(__name__)
# 强制指定静态文件目录，解决Flask静态文件加载问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'upload')
app.config['CLASSIFIED_FOLDER'] = os.path.join(BASE_DIR, 'static', 'classified')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 提升到32MB支持大图片
app.static_folder = os.path.join(BASE_DIR, 'static')
app.static_url_path = '/static'  # 明确静态文件URL路径

# 创建必要文件夹
folders = [
    app.config['UPLOAD_FOLDER'],
    os.path.join(app.config['UPLOAD_FOLDER'], 'single', 'front'),
    os.path.join(app.config['UPLOAD_FOLDER'], 'single', 'back'),
    os.path.join(app.config['UPLOAD_FOLDER'], 'multi'),
    os.path.join(app.config['UPLOAD_FOLDER'], 'multi', 'cropped'),
    app.config['CLASSIFIED_FOLDER']
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ========== 模型加载 ==========
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载ResNet50分类模型（兼容不同输入尺寸）
resnet_model = models.resnet50(pretrained=False)
in_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(in_features, 26)  # 26类钱币
model_path = os.path.join(BASE_DIR, 'model', 'best_ming_coins_model.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在：{model_path}")
resnet_model.load_state_dict(
    torch.load(model_path, map_location=DEVICE)['model_state_dict']
)
resnet_model.to(DEVICE)
resnet_model.eval()

# 加载YOLOv8检测模型
yolov8_path = os.path.join(BASE_DIR, 'model', 'yolov8s-seg.pt')
if not os.path.exists(yolov8_path):
    raise FileNotFoundError(f"YOLO模型文件不存在：{yolov8_path}")
from ultralytics import YOLO

yolov8_model = YOLO(yolov8_path)

# 加载钱币信息
coin_info_path = os.path.join(BASE_DIR, 'coin_info.json')
if not os.path.exists(coin_info_path):
    raise FileNotFoundError(f"钱币信息文件不存在：{coin_info_path}")
with open(coin_info_path, 'r', encoding='utf-8') as f:
    coin_info = json.load(f)

coin_classes = list(coin_info.keys())
for idx, cls in enumerate(coin_classes):
    if cls == "yongchang":  # 若模型把大顺通宝识别为"yongchang"的索引
        coin_classes[idx] = "dashun"  # 强制映射为大顺通宝
# 生成索引->类别映射
idx_to_class = {str(i): cls for i, cls in enumerate(coin_info.keys())}

# ========== 图片预处理（兼容所有尺寸/格式） ==========
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),  # antialias修复尺寸兼容
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#convert('RGB')：将 PNG（透明通道）、GIF（多帧）、BMP 等格式统一转为 RGB，避免模型推理时通道数不匹配；
#强制转为 JPG：统一图片格式，减少后续处理的兼容性问题；
# ========== 工具函数 ==========
def safe_save_image(file, save_path):
    """安全保存图片，兼容所有格式并转换为RGB"""
    try:
        # 打开图片并转换为RGB（兼容gif/bmp/png等）
        img = Image.open(file.stream).convert('RGB')
        # 保存为jpg格式（统一格式，避免兼容问题）
        if not save_path.lower().endswith('.jpg'):
            save_path = os.path.splitext(save_path)[0] + '.jpg'
        img.save(save_path, 'JPEG', quality=95)
        return save_path
    except UnidentifiedImageError:
        return None
    except Exception as e:
        print(f"保存图片失败：{e}")
        return None

#解决 Windows 路径反斜杠问题：URL 中必须用正斜杠，否则浏览器无法加载；
#基于static_folder动态生成相对路径：避免硬编码 URL 导致的路径错误。
def get_static_url(path):
    """生成绝对的静态文件URL，确保图片能加载"""
    if not path or not os.path.exists(path):
        return ""
    # 转换为相对于static的路径
    rel_path = os.path.relpath(path, app.static_folder)
    # 替换反斜杠为正斜杠（兼容Windows）
    rel_path = rel_path.replace(os.sep, '/')
    return f"{app.static_url_path}/{rel_path}"

#softmax：模型输出的是 logits（未归一化），需转换为概率才能表示可信度；
def calculate_confidence(outputs):
    """计算预测可信度（使用softmax转换为概率）"""
    try:
        # 使用softmax计算概率
        probabilities = torch.softmax(outputs, dim=1)
        # 获取最高概率
        max_prob, _ = torch.max(probabilities, 1)
        # 转换为百分比并保留两位小数
        confidence = float(max_prob.item()) * 100
        return round(confidence, 2)
    except Exception as e:
        print(f"计算可信度失败：{e}")
        return 0.0

#手动接管静态文件路由：Flask 默认静态文件路由可能因路径配置、缓存问题失效，手动映射可确保/static/xxx能正确指向静态文件目录；
#send_from_directory：安全地从指定目录返回文件，避免路径遍历漏洞。
# ========== 路由：解决静态文件加载问题 ==========
@app.route('/static/<path:filename>')
def serve_static(filename):
    """手动映射静态文件，确保图片能访问"""
    return send_from_directory(app.static_folder, filename)


# ========== 业务路由 ==========
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/single_coin', methods=['GET', 'POST'])
def single_coin():
    """单钱币识别（修复多格式/尺寸/显示问题）"""
    result = None
    front_img_url = ""  # 前端可直接访问的URL
    back_img_url = ""
    confidence = 0.0  # 识别可信度
    confidence_level = ""  # 可信度等级

    if request.method == 'POST':
        # 处理正面图片
        front_file = request.files.get('front_img')
        if front_file and front_file.filename:
            # 统一扩展名并生成路径
            ext = os.path.splitext(front_file.filename)[1].lower()
            front_filename = f"single_front_{int(time.time())}.jpg"  # 强制jpg
            front_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'single', 'front', front_filename)
            # 安全保存图片
            front_save_path = safe_save_image(front_file, front_save_path)
            if front_save_path:
                front_img_url = get_static_url(front_save_path)

        # 处理背面图片
        back_file = request.files.get('back_img')
        if back_file and back_file.filename:
            back_filename = f"single_back_{int(time.time())}.jpg"
            back_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'single', 'back', back_filename)
            back_save_path = safe_save_image(back_file, back_save_path)
            if back_save_path:
                back_img_url = get_static_url(back_save_path)

        # 模型识别（兼容任意尺寸图片）
        img_path = None
        if front_img_url:
            img_path = os.path.join(app.static_folder, front_img_url.replace(app.static_url_path + '/', ''))
        elif back_img_url:
            img_path = os.path.join(app.static_folder, back_img_url.replace(app.static_url_path + '/', ''))

        if img_path and os.path.exists(img_path):
            try:
                # 加载图片（忽略原始尺寸）
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                # 预测
                with torch.no_grad():
                    outputs = resnet_model(img_tensor)
                    _, pred_idx = torch.max(outputs, 1)
                    pred_class = idx_to_class[str(pred_idx.item())]

                    # 计算可信度
                    confidence = calculate_confidence(outputs)

                    # 确定可信度等级
                    if confidence >= 90:
                        confidence_level = "极高"
                    elif confidence >= 80:
                        confidence_level = "高"
                    elif confidence >= 70:
                        confidence_level = "中等"
                    elif confidence >= 60:
                        confidence_level = "较低"
                    else:
                        confidence_level = "低"

                # 获取结果
                result = coin_info.get(pred_class, {
                    "年代": "未知", "年号": "未知", "出土地点": "未知",
                    "铸地": "未知", "历史地位": "未知", "创建人物": "未知"
                })
                # 添加可信度信息到结果中
                result['可信度'] = f"{confidence}%"
                result['可信度等级'] = confidence_level

                # 分类保存
                class_folder = os.path.join(app.config['CLASSIFIED_FOLDER'], pred_class)
                os.makedirs(class_folder, exist_ok=True)
                if front_save_path:
                    shutil.copy(front_save_path, os.path.join(class_folder, os.path.basename(front_save_path)))
                if back_save_path:
                    shutil.copy(back_save_path, os.path.join(class_folder, os.path.basename(back_save_path)))

            except Exception as e:
                print(f"识别失败：{e}")
                result = {"年号": "识别失败", "年代": "未知", "可信度": "0%", "可信度等级": "无"}

    return render_template(
        'single_coin.html',
        result=result,
        front_img_url=front_img_url,
        back_img_url=back_img_url,
        confidence=confidence,
        confidence_level=confidence_level
    )





#YOLO 检测参数：conf=0.25过滤低置信度的检测框（避免识别无关物体），imgsz=640是 YOLOv8 的推荐输入尺寸；
#坐标校验：裁剪时限制坐标在图片范围内，避免crop函数报错；
#准确率提示：提前告知用户多钱币识别的准确率范围，降低预期偏差；
#逐框处理：对每个检测到的钱币单独裁剪、分类，实现多钱币批量识别。
@app.route('/multi_coin', methods=['GET', 'POST'])
def multi_coin():
    """多钱币识别"""
    results = []
    original_img_url = ""
    cropped_imgs_url = []
    # 多钱币识别准确率提示（固定提示）
    accuracy_note = "多钱币识别由于涉及目标检测+分类两步处理，整体准确率约65-75%，受图片质量、钱币摆放、遮挡等因素影响较大"

    if request.method == 'POST':
        multi_file = request.files.get('multi_img')
        if multi_file and multi_file.filename:
            # 保存原图（统一格式）
            original_filename = f"multi_original_{int(time.time())}.jpg"
            original_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'multi', original_filename)
            original_save_path = safe_save_image(multi_file, original_save_path)
            if original_save_path:
                original_img_url = get_static_url(original_save_path)

                # YOLO检测（兼容不同尺寸）
                img = Image.open(original_save_path).convert('RGB')
                yolo_results = yolov8_model(img, conf=0.25, imgsz=640)  # 自适应尺寸
                yolo_res = yolo_results[0]

                if yolo_res.boxes is not None:
                    for box_idx, box in enumerate(yolo_res.boxes):
                        # 裁剪坐标校验
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img.width, x2)
                        y2 = min(img.height, y2)
                        cropped_img = img.crop((x1, y1, x2, y2)).convert('RGB')

                        # 保存裁剪图
                        cropped_filename = f"multi_crop_{int(time.time())}_{box_idx}.jpg"
                        cropped_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'multi', 'cropped',
                                                         cropped_filename)
                        cropped_img.save(cropped_save_path, 'JPEG')
                        cropped_img_url = get_static_url(cropped_save_path)
                        cropped_imgs_url.append(cropped_img_url)

                        # 分类识别
                        img_tensor = transform(cropped_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            outputs = resnet_model(img_tensor)
                            _, pred_idx = torch.max(outputs, 1)
                            pred_class = idx_to_class[str(pred_idx.item())]

                            # 计算单张裁剪图的可信度
                            confidence = calculate_confidence(outputs)

                            # 确定可信度等级
                            if confidence >= 90:
                                confidence_level = "极高"
                            elif confidence >= 80:
                                confidence_level = "高"
                            elif confidence >= 70:
                                confidence_level = "中等"
                            elif confidence >= 60:
                                confidence_level = "较低"
                            else:
                                confidence_level = "低"

                        # 结果组装
                        coin_detail = coin_info.get(pred_class, {"年号": "未知", "年代": "未知"})
                        coin_detail['img_path'] = cropped_img_url
                        coin_detail['可信度'] = f"{confidence}%"
                        coin_detail['可信度等级'] = confidence_level
                        results.append(coin_detail)

                        # 分类保存
                        class_folder = os.path.join(app.config['CLASSIFIED_FOLDER'], pred_class)
                        os.makedirs(class_folder, exist_ok=True)
                        shutil.copy(cropped_save_path, os.path.join(class_folder, cropped_filename))

    return render_template(
        'multi_coin.html',
        results=results,
        original_img_url=original_img_url,
        cropped_imgs_url=cropped_imgs_url,
        accuracy_note=accuracy_note  # 传递准确率提示
    )




#类别筛选：通过request.args.get('class')实现按类别过滤图片；
#删除逻辑：删除图片后检查文件夹是否为空，空则删除（避免冗余空文件夹）；
#重定向：删除后重定向，避免用户刷新页面导致重复删除；
#仅加载 JPG：过滤非图片文件，避免加载错误。
@app.route('/classified', methods=['GET', 'POST'])
def classified():
    """已分类查询"""
    classes = [d for d in os.listdir(app.config['CLASSIFIED_FOLDER'])
               if os.path.isdir(os.path.join(app.config['CLASSIFIED_FOLDER'], d))]
    selected_class = request.args.get('class', '')

    # 删除图片
    if request.method == 'POST' and request.form.get('action') == 'delete':
        img_url = request.form.get('img_url')
        img_path = os.path.join(app.static_folder, img_url.replace(app.static_url_path + '/', ''))
        if os.path.exists(img_path):
            os.remove(img_path)
            class_folder = os.path.dirname(img_path)
            if os.path.exists(class_folder) and not os.listdir(class_folder):
                os.rmdir(class_folder)
            return redirect(url_for('classified', cla=selected_class))

    # 加载分类图片
    classified_imgs = []
    for cls in classes:
        if selected_class and cls != selected_class:
            continue
        cls_folder = os.path.join(app.config['CLASSIFIED_FOLDER'], cls)
        img_files = [f for f in os.listdir(cls_folder) if f.lower().endswith('.jpg')]
        for img_file in img_files:
            img_path = os.path.join(cls_folder, img_file)
            img_url = get_static_url(img_path)
            classified_imgs.append({
                'img_url': img_url,
                'class_name': cls,
                'coin_info': coin_info.get(cls, {"年号": "未知"})
            })

    return render_template(
        'classified.html',
        classes=classes,
        selected_class=selected_class,
        classified_imgs=classified_imgs,
        coin_info=coin_info
    )


if __name__ == '__main__':
    # 关闭调试模式的缓存，确保图片实时加载
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, host='0.0.0.0', port=5000)