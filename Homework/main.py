import cv2
from matplotlib import pyplot as plt
import numpy as np
from algorithm.ColorRange import ColorRange
from algorithm.BallDetector import BallDetector

def main():
    # 从配置文件加载配置
    config_file = "config/config.json"
    
    #从数据集加载视频并设置保存路径
    video_path = "data/output_rotate2.avi"
    output_path = "output_result2.avi"
    
    #实例化颜色范围对象
    color_range = ColorRange(config_file)
    
    # 实例化检测器（自动从配置文件加载颜色范围和ROI设置）
    detector = BallDetector(video_path,color_range, config_file)
    
    # 处理整个视频
    results = detector.process_video(output_path, show_preview=False)
    
    # 打印处理结果统计
    state_counts = {}
    for result in results:
        state = result["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print("\n检测结果统计:")
    for state, count in state_counts.items():
        print(f"{state}: {count} 帧")

if __name__ == "__main__":
    main()