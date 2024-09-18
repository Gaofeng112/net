import onnxruntime as ort

# 假设我们有三个 ONNX 模型：model1.onnx, model2.onnx, model3.onnx
model_files = ["model1.onnx", "model2.onnx", "model3.onnx"]

# 加载模型并创建会话
sessions = [ort.InferenceSession(model) for model in model_files]

# 假设我们知道第一个模型的输入数据
input_data = {
    "input_tensor": initial_input_data
}

for i, session in enumerate(sessions):
    # 获取当前模型的输入名称
    input_name = session.get_inputs()[0].name

    # 运行推理
    outputs = session.run(None, {input_name: input_data[input_name]})

    # 假设每个模型只有一个输出
    output_name = session.get_outputs()[0].name

    # 保存输出数据作为下一个模型的输入
    if i < len(sessions) - 1:
        next_input_name = sessions[i + 1].get_inputs()[0].name
        input_data[next_input_name] = outputs[0]

# 最终输出结果
final_output = outputs[0]

# 打印最终输出
print(final_output)
input_data = {
    "input_tensor": initial_input_data
}


在处理由多个ONNX子模型组成的大模型时，关键在于管理子模型间的输入和输出，确保数据流正确无误地从一个子模型传递到下一个。以下是一个基于您提供的`subgraphs.txt`文件内容的详细示例，展示如何将一个子模型的输出用作后续子模型的输入。

### 示例：使用`NPUsubgraph0`的输出作为`CPUsubgraph1`的输入

#### 步骤1：加载子模型

首先，加载`NPUsubgraph0`和`CPUsubgraph1`两个子模型。假设这两个模型的ONNX文件分别为`NPUsubgraph0.onnx`和`CPUsubgraph1.onnx`，并且它们位于同一目录下。

```python
# 加载NPUsubgraph0
subgraph0 = onnxruntime.InferenceSession("NPUsubgraph0.onnx", providers=providers)

# 加载CPUsubgraph1
subgraph1 = onnxruntime.InferenceSession("CPUsubgraph1.onnx", providers=providers)
```

#### 步骤2：了解输入和输出

查看`NPUsubgraph0`的输出和`CPUsubgraph1`的输入。

```python
# NPUsubgraph0的输出
outputs_subgraph0 = [out.name for out in subgraph0.get_outputs()]
print("NPUsubgraph0 Outputs:", outputs_subgraph0)

# CPUsubgraph1的输入
inputs_subgraph1 = [inpt.name for inpt in subgraph1.get_inputs()]
print("CPUsubgraph1 Inputs:", inputs_subgraph1)
```

假设`NPUsubgraph0`的输出之一是`/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0`，而`CPUsubgraph1`的一个输入正好是这个输出。

#### 步骤3：准备输入数据

对于`NPUsubgraph0`，我们需要提供其所需的输入数据。假设`sample`和`size`是其输入。

```python
# 准备NPUsubgraph0的输入数据
input_data_subgraph0 = {
    "sample": np.random.randn(1, 4, 32, 32).astype(np.float32),
    "size": np.array([1, 4, 32, 32]).astype(np.int64),
}
```

#### 步骤4：运行NPUsubgraph0

使用`input_data_subgraph0`来运行`NPUsubgraph0`。

```python
# 运行NPUsubgraph0
outputs_subgraph0 = subgraph0.run(None, input_data_subgraph0)
```

`outputs_subgraph0`现在是一个列表，其中包含了`NPUsubgraph0`的所有输出。

#### 步骤5：选择输出并传递给CPUsubgraph1

从`outputs_subgraph0`中选择正确的输出，并将其传递给`CPUsubgraph1`。

```python
# 假设我们需要的输出是列表中的第一个元素
output_for_subgraph1 = outputs_subgraph0[0]

# 准备CPUsubgraph1的输入数据
input_data_subgraph1 = {
    "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0": output_for_subgraph1,
    # 如果有其他输入，也在这里添加
}

# 运行CPUsubgraph1
outputs_subgraph1 = subgraph1.run(None, input_data_subgraph1)
```

### 总结

这个过程需要对每个子模型的输入和输出有清晰的认识，并且在每个子模型执行后正确地传递输出数据给下一个子模型。为了自动化这个过程，你可以编写一个函数或类，该函数或类能够遍历所有子模型，自动处理输入输出的传递，并执行整个流水线。

在处理更复杂的子模型网络时，你可能需要维护一个字典或数据结构来跟踪所有中间结果，这样可以轻松地查找和传递数据。此外，你可能还需要处理数据类型转换或重塑等问题，以确保数据格式符合每个子模型的要求。