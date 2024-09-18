import os

def delete_quantized_files(folder_path):
    # 列出指定文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历所有文件
    for file in files:
        # 检查文件是否是以 _quant 结尾的 .onnx 或 .tflite 文件
        if file.endswith('_quant.onnx') or file.endswith('_quant.tflite'):
            file_path = os.path.join(folder_path, file)
            # 删除文件
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# 主执行块
if __name__ == "__main__":
    folder_path = '../onnx_lib'  # 包含模型的目录路径
    
    delete_quantized_files(folder_path)  # 删除后缀含有 _quant 的 .onnx 和 .tflite 文件
