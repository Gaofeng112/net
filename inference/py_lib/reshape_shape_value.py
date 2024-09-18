import onnx
import numpy as np
from onnx import helper

# 加载 ONNX 模型
model_path = "./documents-export-2024-7-30/npusubgraph0.onnx"
model = onnx.load(model_path)

# 更新形状张量的值
new_shape = np.array([1, 32, -1])  # 新的形状张量值
new_shape_data = new_shape.flatten().tolist()

# 查找形状张量
for tensor in model.graph.initializer:
    if tensor.name == "down_blocks0resnets0norm1Constant_output_0":  # 根据实际情况更改此名称
        break
else:
    raise ValueError(f"Could not find initializer with name '{'down_blocks0resnets0norm1Constant_output_0'}")

# 更新张量数据
new_shape_tensor = helper.make_tensor(name=tensor.name, data_type=onnx.TensorProto.INT64, dims=list(tensor.dims), vals=new_shape_data)

# 添加新的形状张量到模型中
model.graph.initializer.append(new_shape_tensor)

# 删除旧的形状张量
model.graph.initializer.remove(tensor)

# 保存更新后的模型
onnx.save(model, "npusubgraph0_v1.onnx")

# import onnx
# import numpy as np
# from onnx import helper

# # 加载 ONNX 模型
# model_path = "./documents-export-2024-7-30/npusubgraph0.onnx"
# model = onnx.load(model_path)

# # 更新形状张量的值
# new_shape = np.array([1, 32, -1])  # 新的形状张量值
# new_shape_data = new_shape.flatten().tolist()

# # 查找形状张量
# for tensor in model.graph.initializer:
#     if tensor.name == "down_blocks0resnets0norm1Constant_output_0":  # 根据实际情况更改此名称
#         break
# else:
#     raise ValueError(f"Could not find initializer with name '{'down_blocks0resnets0norm1Constant_output_0'}")

# # 更新张量数据
# new_shape_tensor = helper.make_tensor(name=tensor.name, data_type=onnx.TensorProto.INT64, dims=list(tensor.dims), vals=new_shape_data)

# # 保存更新后的模型
# onnx.save(model, "npusubgraph0_v1.onnx")

# import onnx
# from onnx import numpy_helper
# import numpy as np

# # 加载 ONNX 模型
# model_path = "./documents-export-2024-7-30/npusubgraph0.onnx"
# model = onnx.load(model_path)

# # 获取模型的图结构
# graph = model.graph

# # 遍历所有节点
# for node in graph.node:
#     # 检查节点是否为 Reshape
#     if node.op_type == 'Reshape':
#         # 获取 Reshape 节点的输入和输出名称
#         input_names = node.input
#         output_names = node.output

#         # 找到输入张量的定义
#         input_tensor_def = next((t for t in list(graph.initializer) + list(graph.value_info) if t.name == input_names[0]), None)
#         shape_tensor_def = next((t for t in list(graph.initializer) + list(graph.value_info) if t.name == input_names[1]), None)

#         # 尝试获取输入张量的值
#         input_tensor_value = None
#         if input_tensor_def and input_tensor_def in graph.initializer:
#             input_tensor_value = numpy_helper.to_array(input_tensor_def)

#         # 尝试获取形状张量的值
#         shape_tensor_value = None
#         if shape_tensor_def and shape_tensor_def in graph.initializer:
#             shape_tensor_value = numpy_helper.to_array(shape_tensor_def)

#         # 打印相关信息
#         print(f"Reshape Node: {node.name}")
#         print(f"  Input Tensor Name: {input_names[0]}, Shape: {input_tensor_value.shape if input_tensor_value is not None else 'Not found'}")
#         print(f"  Shape Tensor Name: {input_names[1]}, Value: {shape_tensor_value}")
#         print(f"  Output Tensor Name: {output_names[0]}")

#         # 如果需要进一步处理，比如使用 TensorFlow 进行 reshape 操作
#         if input_tensor_value is not None and shape_tensor_value is not None:
#             # 使用 NumPy 计算最终的形状
#             final_shape = calculate_reshape_shape(input_tensor_value.shape, shape_tensor_value)
#             print(f"  Final Shape: {final_shape}")

# def calculate_reshape_shape(input_shape, target_shape):
#     total_elements = np.prod(input_shape)
#     print("input_shape:", input_shape, "total_elements:", total_elements)
#     fixed_elements = np.prod([dim for dim in target_shape if dim > 0])
#     print(target_shape, fixed_elements)
#     if -1 in target_shape:
#         dynamic_dim_size = total_elements // fixed_elements
#         final_shape = [dim if dim != -1 else dynamic_dim_size for dim in target_shape]
#         final_shape = [dim if dim != 0 else 1 for dim in final_shape]
#         print("final_shape:", final_shape, "dynamic_dim_size:", dynamic_dim_size)
#         return final_shape
#     else:
#         return target_shape

