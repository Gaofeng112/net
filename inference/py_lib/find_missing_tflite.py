import os

def find_missing_tflite_files(folder_path, output_file_path):
    # 创建一个字典来存储 onnx 文件名及其对应的 tflite 文件名
    onnx_files = {}
    tflite_files = set()
    
    # 列出文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历所有文件
    for file in files:
        if file.endswith('.onnx'):
            # 提取 onnx 文件的基本名称
            base_name = os.path.splitext(file)[0]
            onnx_files[base_name] = file
        elif file.endswith('.tflite'):
            # 提取 tflite 文件的基本名称
            base_name = os.path.splitext(file)[0]
            tflite_files.add(base_name)

    # 找出缺少 tflite 文件的 onnx 文件
    missing_files = []
    for base_name, onnx_file in onnx_files.items():
        if base_name not in tflite_files:
            missing_files.append(onnx_file)

    # 将缺少 tflite 文件的 onnx 文件名写入输出文件
    with open(output_file_path, 'w') as f:
        for file in missing_files:
            f.write(file + '\n')

    print(f"Missing TFLite files written to {output_file_path}")

# 主执行块
if __name__ == "__main__":
    folder_path = '../lib_onnx_tflite'  # 包含模型的目录路径
    output_file_path = './missing_tflite_files.txt'  # 输出文件路径
    
    find_missing_tflite_files(folder_path, output_file_path)