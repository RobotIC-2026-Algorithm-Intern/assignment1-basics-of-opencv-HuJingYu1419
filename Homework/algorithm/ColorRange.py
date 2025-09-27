import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
import os

class ColorRange:
    """
    管理不同颜色的HSV阈值范围，用于颜色识别和掩膜生成
    """
    
    def __init__(self, config_file=None):
        """
        构造函数，初始化颜色范围字典
        格式: {颜色名称: [(lower1, upper1), (lower2, upper2), ...]}
        
        参数:
            config_file: 配置文件路径，如果提供则从文件加载颜色范围
        """
        self.color_ranges = {}
        
        if config_file and os.path.exists(config_file):
            self.load_from_config(config_file)
    
    def load_from_config(self, config_file):
        """
        从配置文件加载颜色范围
        
        参数:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'color_ranges' in config:
                self.color_ranges = {}
                for color_name, ranges in config['color_ranges'].items():
                    self.color_ranges[color_name] = []
                    for range_pair in ranges:
                        lower = tuple(range_pair[0])
                        upper = tuple(range_pair[1])
                        self.color_ranges[color_name].append((lower, upper))
                
                print(f"从配置文件 {config_file} 加载了 {len(self.color_ranges)} 种颜色范围")
                
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def add_color_range(self, color_name, lower, upper):
        """
        添加颜色范围的方法
        
        参数:
            color_name: 颜色名称，如 "red", "blue"
            lower: HSV下界阈值，格式 (H, S, V)
            upper: HSV上界阈值，格式 (H, S, V)
        """
        if color_name not in self.color_ranges:
            self.color_ranges[color_name] = []
        
        # 将numpy数组转换为元组（如果输入是numpy数组）
        lower_tuple = tuple(lower) if isinstance(lower, np.ndarray) else lower
        upper_tuple = tuple(upper) if isinstance(upper, np.ndarray) else upper
        
        self.color_ranges[color_name].append((lower_tuple, upper_tuple))
    
    def get_mask(self, hsv_image, color_name):
        """
        生成指定颜色的掩膜
        
        参数:
            hsv_image: HSV格式的图像
            color_name: 颜色名称
            
        返回:
            mask: 二值掩膜，白色区域表示符合颜色范围
        """
        if color_name not in self.color_ranges:
            raise ValueError(f"颜色 '{color_name}' 未定义")
        
        # 创建空掩膜
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        # 合并该颜色的所有范围
        for lower, upper in self.color_ranges[color_name]:
            lower_array = np.array(lower, dtype=np.uint8)
            upper_array = np.array(upper, dtype=np.uint8)
            
            color_mask = cv2.inRange(hsv_image, lower_array, upper_array)
            mask = cv2.bitwise_or(mask, color_mask)
        
        return mask
    
    def get_combined_mask(self, hsv_image, color_list):
        """
        生成多个颜色的组合掩膜
        
        参数:
            hsv_image: HSV格式的图像
            color_list: 颜色名称列表，如 ["red", "blue"]
            
        返回:
            combined_mask: 多个颜色的合并掩膜
        """
        if not color_list:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for color_name in color_list:
            if color_name not in self.color_ranges:
                raise ValueError(f"颜色 '{color_name}' 未定义")
            
            color_mask = self.get_mask(hsv_image, color_name)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        return combined_mask
    
    def get_color_names(self):
        """
        获取已定义的所有颜色名称
        
        返回:
            list: 颜色名称列表
        """
        return list(self.color_ranges.keys())
    
    def remove_color_range(self, color_name):
        """
        移除指定颜色的所有范围
        
        参数:
            color_name: 要移除的颜色名称
        """
        if color_name in self.color_ranges:
            del self.color_ranges[color_name]
    
    def clear_all(self):
        """
        清空所有颜色范围
        """
        self.color_ranges.clear()

def show_img(*imgs: np.ndarray) -> None:
    """使用matplotlib在Notebook中绘制图像"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.figure()
    color=["red","blue","purple","muti"]
    for idx, img in enumerate(imgs):
        plt.subplot(1, len(imgs), idx + 1)
        plt.title(f"{color[idx]}")
        if len(img.shape) == 2:
            # 灰度图
            plt.imshow(img, cmap='gray')
        elif len(img.shape) == 3:
            # 彩色图在绘制时需要将OpenCV的BGR格式转为RGB格式
            plt.imshow(img[:, :, ::-1]) # 对[:, :, ::-1]有疑惑的同学可以研究一下Python的切片
    plt.show()

if __name__ == "__main__":
    
    # 从配置文件加载颜色范围
    color_mgr = ColorRange("config/config.json")

    # 使用示例
    img=cv2.imread("data/output1.jpg")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 生成红色掩膜（自动合并两个区间）
    red_mask = color_mgr.get_mask(hsv_img, "red")
    blue_mask= color_mgr.get_mask(hsv_img,"blue")
    purple_mask= color_mgr.get_mask(hsv_img,"purple")

    # 生成红蓝组合掩膜
    multi_mask = color_mgr.get_combined_mask(hsv_img, ["red", "blue"])
    
    # 获取所有已定义的颜色
    print("已定义颜色:", color_mgr.get_color_names())
    
    show_img(red_mask,blue_mask,purple_mask,multi_mask)