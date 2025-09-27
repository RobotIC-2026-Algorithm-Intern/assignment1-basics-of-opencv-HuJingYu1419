import cv2
import numpy as np

def process_video(input_path, output_image_path, output_video_path):
    """
    读取视频，将每帧倒转180度，保存第一帧为图片，并输出倒转后的视频
    """
    # 创建视频捕获对象
    cap = cv2.VideoCapture(input_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    first_frame_saved = False
    
    print("开始处理视频...")
    
    while True:
        # 读取一帧
        success, frame = cap.read()
        
        if not success:
            break
        
        # 将帧倒转180度
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # 如果是第一帧，保存为图片
        if not first_frame_saved:
            cv2.imwrite(output_image_path, rotated_frame)
            print(f"第一帧已保存为: {output_image_path}")
            first_frame_saved = True
        
        # 写入倒转后的帧到输出视频
        out.write(rotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧显示一次进度
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"视频处理完成！")
    print(f"总共处理了 {frame_count} 帧")
    print(f"倒转后的视频已保存为: {output_video_path}")
    
    return True

# 使用示例
if __name__ == "__main__":
    input_video = r"res\output1.avi"        # 输入视频路径
    output_image = r"res\output2.jpg" # 输出的第一帧图片
    output_video = r"res\output_rotate2.avi"     # 输出的倒转视频
    
    process_video(input_video, output_image, output_video)