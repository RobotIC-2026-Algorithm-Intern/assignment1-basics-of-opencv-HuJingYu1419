import matplotlib.pyplot as plt
import cv2

def display_image_cv2(image_path, figsize=(10, 8)):
    """
    使用OpenCV读取图片并用matplotlib显示
    """
    # 读取图像（OpenCV读取的是BGR格式）
    img = cv2.imread(image_path)
    
    if img is None:
        print("无法读取图像，请检查文件路径")
        return
    
    # 将BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb=img_rgb[80:320,100:500,:] #范围测试
    
    # 显示图像
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.title(f'test: {image_path}')
    plt.show()
    
    print(f"图像形状: {img.shape}")
    print(f"数据类型: {img.dtype}")

# 使用示例
image_path = "Homework/data/output1.jpg"  # 替换为你的图片路径
display_image_cv2(image_path)