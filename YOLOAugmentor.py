import cv2
import numpy as np
import os
import json  # 新增json库导入
from data_aug.data_aug import *
from data_aug.bbox_util import *
from toolz import curry
import shutil  # 新增导入
class YOLOAugmentor:
    def __init__(self, img_dir, label_dir, output_dir, class_mapping=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.class_mapping = class_mapping or {}  # 新增类别映射字典
        os.makedirs(output_dir, exist_ok=True)
        
    def _yolo_to_abs(self, labels, img_w, img_h):
        """将YOLO格式转换为绝对坐标(x1,y1,x2,y2)"""
        bboxes = []
        for label in labels:
            class_id, cx, cy, w, h = map(float, label.split())
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            bboxes.append([x1, y1, x2, y2, int(class_id)])
        return np.array(bboxes)
    
    def _abs_to_yolo(self, bboxes, img_w, img_h):
        """将绝对坐标转换回YOLO格式"""
        labels = []
        for box in bboxes:
            x1, y1, x2, y2, class_id = box
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            cx = ((x1 + x2)/2) / img_w
            cy = ((y1 + y2)/2) / img_h
            labels.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return labels
    
    def _load_data(self, img_name):
        """加载图片和对应标签（支持txt/json）"""
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        # 查找标签文件（优先json格式）
        label_path = os.path.join(self.label_dir, f"{base_name}.json")
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)[:,:,::-1]
        h, w = img.shape[:2]
        
        if label_path.endswith('.json'):
            with open(label_path) as f:
                data = json.load(f)
            bboxes = self._parse_json_labels(data, w, h)
        else:
            with open(label_path) as f:
                labels = f.read().splitlines()
            bboxes = self._yolo_to_abs(labels, w, h)
            
        return img, bboxes, (w, h)

    def _parse_json_labels(self, data, img_w, img_h):
        """解析Labelme格式的JSON标注"""
        bboxes = []
        json_w = data['imageWidth']
        json_h = data['imageHeight']
        
        for shape in data['shapes']:
            # 转换多边形为矩形边界框
            points = np.array(shape['points'])
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            
            # 坐标缩放适应实际图像尺寸
            x1 = x1 * img_w / json_w
            y1 = y1 * img_h / json_h
            x2 = x2 * img_w / json_w
            y2 = y2 * img_h / json_h
            
            # 修改类别ID转换部分
            class_name = shape['label']
            if class_name in self.class_mapping:
                class_id = self.class_mapping[class_name]
            else:
                try:  # 尝试转换为数字ID
                    class_id = int(class_name)
                except ValueError:
                    raise ValueError(f"类别 '{class_name}' 未在class_mapping中定义且不是数字ID，请检查输入标签")

            bboxes.append([x1, y1, x2, y2, class_id])
            
        return np.array(bboxes)
    
    def _save_data(self, img, bboxes, img_info, img_name):
        """保存增强后的结果"""
        w, h = img_info
        labels = self._abs_to_yolo(bboxes, w, h)
        
        # 保存图片
        cv2.imwrite(os.path.join(self.output_dir, img_name), img[:,:,::-1])
        # 保存标签
        with open(os.path.join(self.output_dir, img_name.replace('.jpg', '.txt')), 'w') as f:
            f.write('\n'.join(labels))
    
    @curry
    def horizontal_flip(self, p,  img, bboxes):
        return RandomHorizontalFlip(p=p)(img, bboxes)
    
    @curry
    def random_scale(self, low, high, img, bboxes):  # 修改参数为范围值
        scale = np.random.uniform(low, high)
        return RandomScale(scale, diff=True)(img, bboxes)

    @curry
    def scale(self, low, high, img, bboxes):  # 修改参数为范围值
        scale = np.random.uniform(low, high)
        return Scale(scale_x=scale, scale_y=scale)(img, bboxes)
    
    @curry
    def random_rotate(self, low, high, img, bboxes):
        angle = np.random.uniform(low, high)
        # 修改为传入角度范围元组
        return RandomRotate(angle=(low, high))(img, bboxes)
    
    @curry
    def random_translate(self, x_range, y_range, img, bboxes):  # 修改为范围元组
        tx = np.random.uniform(x_range[0], x_range[1])
        ty = np.random.uniform(y_range[0], y_range[1])
        return Translate(tx, ty)(img, bboxes)

        
    def _showOutput(self, img, bboxes):
        """显示增强后的结果"""
        # 手动绘制边界框
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Augmented Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showFirstOutput(self):
        """读取output目录下第一个图片，并读取对应的txt，绘制后展示，按n键显示下一张图片"""
        output_files = os.listdir(self.output_dir)
        image_files = [f for f in output_files if f.endswith('.jpg')]
        if not image_files:
            print("No image files found in the output directory.")
            return

        index = 0
        cv2.namedWindow('Augmented Image', cv2.WINDOW_NORMAL)  # 创建可重用的窗口
        
        while index < len(image_files):
            # 先销毁之前的窗口内容
            cv2.imshow('Augmented Image', np.zeros((1,1,3), np.uint8))
            
            image_name = image_files[index]
            image_path = os.path.join(self.output_dir, image_name)
            label_path = os.path.join(self.output_dir, image_name.replace('.jpg', '.txt'))

            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            with open(label_path) as f:
                labels = f.read().splitlines()
            bboxes = self._yolo_to_abs(labels, w, h)

            # 修改后的显示逻辑
            display_img = img.copy()
            for box in bboxes:
                print(box)
                x1, y1, x2, y2, cls = map(int, box)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 翻转 class_mapping 的键值对
                reversed_class_mapping = {v: k for k, v in self.class_mapping.items()}
                cv2.putText(display_img, str(reversed_class_mapping.get(cls, cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imshow('Augmented Image', display_img)
            key = cv2.waitKey(0)
            
            if key == ord('n'):  # 按n键显示下一张
                index += 1
            elif key == 27:  # ESC键退出
                break
                
            cv2.destroyAllWindows()
            
    def process(self, aug_sequence, num_augments):
        """处理整个数据集，生成指定数量的增强图片"""
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.jpg'):
                original_img, original_bboxes, img_info = self._load_data(img_name)
                
                for i in range(num_augments):
                    # 每次增强前重置随机种子
                    np.random.seed()  # 使用系统时间作为种子
                    random.seed()     # 重置Python内置随机种子
                    
                    img = original_img.copy()
                    bboxes = original_bboxes.copy()
                    
                    # 应用数据增强序列时会生成不同的随机参数
                    for aug_func in aug_sequence:
                        img, bboxes = aug_func(img, bboxes)
                    
                    base_name = os.path.splitext(img_name)[0]
                    new_name = f"{base_name}_aug_{i}.jpg"
                    self._save_data(img, bboxes, img_info, new_name)

    def collect(self, train=0.7, val=0.2, test=0.1):
        """将output目录数据按比例分配到train/val/test子目录"""
        # 验证比例有效性
        assert abs(train + val + test - 1.0) < 1e-6, "比例总和必须等于1"
        assert train > 0 and val > 0 and test > 0, "所有比例必须大于0"
        
        # 获取所有图片文件并打乱
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
        np.random.shuffle(all_files)
        total = len(all_files)
        
        # 计算各数据集数量
        train_num = int(np.floor(total * train))
        val_num = int(np.floor(total * val))
        test_num = total - train_num - val_num
        
        # 创建子目录
        subsets = {
            'train': (0, train_num),
            'val': (train_num, train_num + val_num),
            'test': (train_num + val_num, total)
        }
        
        for subset in subsets:
            subset_dir = os.path.join(self.output_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            # 创建 images 和 labels 子目录
            images_dir = os.path.join(subset_dir, 'images')
            labels_dir = os.path.join(subset_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # 移动图片和标签文件
            start, end = subsets[subset]
            for fname in all_files[start:end]:
                # 移动图片
                src_img = os.path.join(self.output_dir, fname)
                dst_img = os.path.join(images_dir, fname)
                shutil.move(src_img, dst_img)
                
                # 移动标签
                txt_name = fname.replace('.jpg', '.txt')
                src_txt = os.path.join(self.output_dir, txt_name)
                dst_txt = os.path.join(labels_dir, txt_name)
                if os.path.exists(src_txt):
                    shutil.move(src_txt, dst_txt)

if __name__ == "__main__":
    augmentor = YOLOAugmentor(
        img_dir=r"C:\Users\m1876\Desktop\project\DataAugmentationForObjectDetection",
        label_dir=r"C:\Users\m1876\Desktop\project\DataAugmentationForObjectDetection",
        output_dir=r"C:\Users\m1876\Desktop\project\DataAugmentationForObjectDetection\output",
        class_mapping={'red_blood_bar': 0, 'red_blood_bar_t': 1}
    )

    # 修正后的增强序列定义
    aug_sequence = [
        augmentor.horizontal_flip(0.7),
        augmentor.scale(-0.1, 0.1),  # 直接传递范围参数
        augmentor.random_rotate(-5, 5),     # 直接传递范围参数
        augmentor.random_translate((0, 0.3), (0, 0.3))  # 使用元组指定范围
    ]

    # 执行增强处理（生成100张）
    augmentor.process(aug_sequence, 10)
    augmentor.showFirstOutput()
    # augmentor.collect(train=0.7, val=0.2, test=0.1)