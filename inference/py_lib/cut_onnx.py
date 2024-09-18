
import onnx
import re
print("python executed")
# extract_onnx_lib.split_onnx('npuCutInstruction.txt','npu')
# extract_onnx_lib.split_onnx('cpuCutInstruction.txt','cpu')
#extract_onnx_lib.split_onnx('cpuCutInstructionlast.txt','cpu2')

# extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt')


# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_matmul.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/Reshape:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/MatMul:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_reshape.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/MatMul:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_add.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/BiasAdd:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

input_path='./unet_32_sim.onnx'
output_path='./matmul_four.onnx'
input_names=['/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_output_0','/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Mul_1_output_0']
output_names=['/down_blocks.0/attentions.0/transformer_blocks.0/attn1/MatMul_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)
