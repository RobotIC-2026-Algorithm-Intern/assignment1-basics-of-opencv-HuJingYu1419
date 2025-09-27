import cv2
import numpy as np
import json
import os
from .ColorRange import ColorRange

class BallDetector:
    """
    球检测器主类，整合视频读取和球检测逻辑
    """
    
    def __init__(self, video_path=None, color_range=None,config_file=None):
        """
        构造函数
        
        参数:
            video_path: 视频文件路径，如果为None则稍后设置
            config_file: 配置文件路径，如果提供则从文件加载配置
        """
        self.video_path = video_path
        self.cap = None
        self.color_range = color_range
        self.roi = None  # ROI区域格式: (y_start, y_end, x_start, x_end)
        self.pixel_threshold = 500  # 默认阈值
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """
        从配置文件加载配置
        
        参数:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载颜色范围
            if 'color_ranges' in config:
                for color_name, ranges in config['color_ranges'].items():
                    for range_pair in ranges:
                        lower = tuple(range_pair[0])
                        upper = tuple(range_pair[1])
                        self.color_range.add_color_range(color_name, lower, upper)
            
            # 加载ROI设置
            if 'roi' in config:
                roi_config = config['roi']
                self.set_roi(
                    roi_config.get('y_start', 0),
                    roi_config.get('y_end', 0),
                    roi_config.get('x_start', 0),
                    roi_config.get('x_end', 0)
                )
            
            # 加载检测阈值
            if 'detection_threshold' in config:
                self.pixel_threshold = config['detection_threshold']
            
            print(f"从配置文件 {config_file} 加载配置成功")
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def set_color_ranges(self, color_name, lower, upper):
        """添加颜色阈值"""
        self.color_range.add_color_range(color_name, lower, upper)
        
    
    def set_roi(self, y_start, y_end, x_start, x_end):
        """
        设置ROI区域（感兴趣区域）
        
        参数:
            y_start, y_end: Y轴起始和结束位置
            x_start, x_end: X轴起始和结束位置
        """
        if y_start == 0 and y_end == 0 and x_start == 0 and x_end == 0:
            self.roi = None  # 如果都是0，则清除ROI
        else:
            self.roi = (y_start, y_end, x_start, x_end)
    
    def clear_roi(self):
        """清除ROI设置，处理整个图像"""
        self.roi = None
    
    
    def process_frame(self, frame, display_result=True, pixel_threshold=None):
        """
        处理单帧图像，检测球的状态
        
        参数:
            frame: 输入图像帧（BGR格式）
            display_result: 是否在图像上显示结果
            pixel_threshold: 像素数量阈值，如果为None则使用类默认值
            
        返回:
            result: 检测结果字典
            display_frame: 处理后的图像（如果display_result=True）
        """
        # 使用默认阈值或参数阈值
        if pixel_threshold is None:
            pixel_threshold = self.pixel_threshold
        
        # 备份原图用于显示
        display_frame = frame.copy() if display_result else None
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测每种颜色的像素数量
        colors = ["red", "blue", "purple"]
        color_pixel_counts = {}
        color_masks = {}
        
        for color in colors:
            # 生成颜色掩膜
            mask = self.color_range.get_mask(hsv, color)
            
            # 应用ROI区域
            if self.roi is not None:
                y_start, y_end, x_start, x_end = self.roi
                roi_mask = np.zeros_like(mask)
                roi_mask[y_start:y_end, x_start:x_end] = 1
                mask = cv2.bitwise_and(mask, mask, mask=roi_mask)
            
            # 统计该颜色的像素数量
            pixel_count = np.sum(mask > 0)
            color_pixel_counts[color] = pixel_count
            color_masks[color] = mask
        
        # 确定检测结果
        ball_state = "No Ball"
        detected_color = None
        max_pixels = 0
        
        for color, count in color_pixel_counts.items():
            if count > pixel_threshold and count > max_pixels:
                max_pixels = count
                detected_color = color
        
        if detected_color:
            ball_state = {"red": "Red Ball", "blue": "Blue Ball", "purple": "Purple Ball"}[detected_color]
        
        # 准备返回结果
        result = {
            "state": ball_state,
            "color": detected_color,
            "pixel_counts": color_pixel_counts,
            "max_pixels": max_pixels
        }
        
        # 如果需要显示结果
        if display_result:
            # 创建掩膜可视化图像
            mask_visualization = np.zeros_like(frame)
            
            if detected_color:
                # 只显示检测到的颜色的掩膜区域
                detected_mask = color_masks[detected_color]
                
                # 将原图中掩膜区域外的部分设为黑色
                masked_frame = frame.copy()
                for i in range(3):  # 对BGR三个通道分别处理
                    masked_frame[:, :, i] = cv2.bitwise_and(
                        masked_frame[:, :, i], 
                        masked_frame[:, :, i], 
                        mask=detected_mask
                    )
                
                # 将处理后的图像叠加到可视化中
                mask_visualization = masked_frame
                
                # 显示文本颜色根据检测到的球颜色变化
                text_color = {
                    "red": (0, 0, 255),     # 红色 - BGR
                    "blue": (255, 0, 0),    # 蓝色 - BGR  
                    "purple": (255, 0, 255) # 紫色 - BGR
                }[detected_color]
            else:
                # 未检测到球，显示原图但较暗
                mask_visualization = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
                text_color = (255, 255, 255)  # 红色文字表示未检测到
            
            # 在左上角显示检测结果（只显示一行）
            text = ball_state
            
            # 添加背景框使文字更清晰
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x, text_y = 20, 50
            
            # 绘制半透明背景
            overlay = mask_visualization.copy()
            cv2.rectangle(overlay, (text_x-10, text_y-text_size[1]-10), 
                        (text_x + text_size[0] + 10, text_y + 10), 
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, mask_visualization, 0.4, 0, mask_visualization)
            
            # 绘制文字
            cv2.putText(mask_visualization, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            display_frame = mask_visualization
        
        return result, display_frame
    
    def process_video(self, output_path=None, show_preview=False, pixel_threshold=None):
        """
        处理整个视频
        
        参数:
            output_path: 输出视频路径，如果为None则不保存
            show_preview: 是否显示实时预览
            pixel_threshold: 像素数量阈值，如果为None则使用类默认值
            
        返回:
            results: 每帧的检测结果列表
        """
        if self.video_path is None:
            raise ValueError("未设置视频路径")
        
        # 使用默认阈值或参数阈值
        if pixel_threshold is None:
            pixel_threshold = self.pixel_threshold
        
        # 打开视频
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频信息
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        print(f"开始处理视频: {self.video_path}")
        print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")
        print(f"检测阈值: {pixel_threshold}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 处理当前帧
            result, display_frame = self.process_frame(frame, display_result=True, pixel_threshold=pixel_threshold)
            results.append(result)
            
            # 显示进度
            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧显示一次进度
                print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # 显示预览
            if show_preview:
                cv2.imshow('Ball Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 写入输出视频
            if writer and display_frame is not None:
                writer.write(display_frame)
        
        # 清理资源
        if self.cap:
            self.cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
        return results
    
    def __del__(self):
        """析构函数，确保资源释放"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 从配置文件加载配置
    color_range=ColorRange()
    detector = BallDetector("data/output_rotate1.avi",color_range, "config/config.json")
    
    # 处理整个视频
    results = detector.process_video(output_path="output_result1.avi", show_preview=False)