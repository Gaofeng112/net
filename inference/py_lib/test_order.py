import re

# 读取文件
with open('../subgraphs_tflite_0816.txt', 'r') as file:
    content = file.read()

# 创建一个字典来存储子图的order和对应的文件路径
subgraph_order_map = {}

# 正则表达式匹配所有子图的order和名称--保存为元组，内容分别为3个捕获项
# 改进正则表达式以便更准确地匹配
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

# 构建子图的文件路径并存储到字典中
for match in matches:
    subgraph_type, subgraph_number, order = match
    # 将NPU和CPU转换为小写
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"../subgraphs_0816/{lower_subgraph_type}subgraph{subgraph_number}.tflite"
    # 如果order已经存在，则将路径添加到一个列表中，否则创建一个新的列表
    if int(order) in subgraph_order_map:
        subgraph_order_map[int(order)].append(file_path)
    else:
        subgraph_order_map[int(order)] = [file_path]

# 按照order排序子图文件路径
sorted_file_paths = []
for order in sorted(subgraph_order_map.keys()):
    sorted_file_paths.extend(subgraph_order_map[order])
    print(f"order{order}:{subgraph_order_map[order]}")

# 打印排序后的文件路径列表
# print(sorted_file_paths)