from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel
import os
import json
import pandas as pd
from utils import md_to_dict_str
import torch
from datasets import Dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置可见GPU，如果需要的话

model_id = os.getenv("MODEL_ID", "/data/download-model/gpt-oss-20b")  # 基础模型ID或路径
final_adapter_path = os.getenv("FINAL_ADAPTER_PATH",
                               "/data/finetuning/finetuned-model/gpt-oss-20b-bid-adapter")  # LoRA适配器保存路径
final_merged_path = os.getenv("FINAL_MERGED_PATH",
                              "/data/finetuning/finetuned-model/gpt-oss-20b-bid")  # 合并后的模型保存路径
EPOCHS = int(os.getenv("EPOCHS", 2))
LR = float(os.getenv("LR", 2e-5))
tokenizer = AutoTokenizer.from_pretrained(model_id)  # 加载tokenizer

dataset_id = './dataset/bid-announcement-zh-v1.0.jsonl'  # 数据集路径

dataset = pd.read_json(dataset_id, lines=True)  # 读取JSONL格式的数据集

dataset['output'] = dataset['output'].apply(md_to_dict_str)  # 提取output字段中的字典字符串
print(f"output示例: {dataset.loc[0, 'output']}")
gt_path = "./data/gt.txt"
if not os.path.exists(gt_path):
    # 文件不存在，创建并写入内容
    with open(gt_path, "w", encoding="utf-8") as f:
        for row in dataset['output']:
            # 解析JSON并重新序列化，确保中文正常显示
            f.write(json.dumps(json.loads(row), ensure_ascii=False) + '\n')
    print(f"文件已创建并写入: {gt_path}")
else:
    print(f"文件已存在，不执行操作: {gt_path}")
print(f"dataset length: {len(dataset)}")  # 打印数据集长度
dataset = dataset[dataset['output'] != '']  # 移除output为空的行
print(f"output示例: {dataset.loc[0, 'output']}")
print(f"dataset length after drop '': {len(dataset)}")  # 打印过滤后的数据集长度


def formatting(example: dict) -> str:
    """
    格式化数据为聊天模型可接受的格式
    :param example: 数据示例
    :return: 格式化后的字符串
    """
    messages = [
        {"role": "system", "content": example['instruction']},  # 定义developer角色和内容
        {"role": "user", "content": example['input'][:4096]},  # 定义user角色和内容
        {"role": "assistant", "content": example['output']}  # 定义assistant角色和内容
    ]
    return tokenizer.apply_chat_template(  # 使用tokenizer的chat_template函数格式化消息
        messages,
        tokenize=False,  # 不进行tokenize，返回字符串
        add_generation_prompt=False,  # 不添加生成提示
        enable_thinking=False,
    )


dataset['text'] = dataset.apply(
    lambda x: formatting(x),
    axis=1
)  # 将formatting函数应用于数据集的每一行
dataset = dataset[['text']]  # 只保留text列
if dataset.loc[0, 'text'].find("Reasoning: medium"):
    dataset['text'] = dataset['text'].apply(
        lambda x: x.replace("Reasoning: medium", "Reasoning: low")
    )
print(f"数据示例\n{dataset.loc[0, 'text']}")  # 打印数据集中的一个示例

dataset = Dataset.from_pandas(dataset)  # 将pandas DataFrame转换为datasets Dataset对象
print(dataset)  # 打印Dataset对象

model = AutoModelForCausalLM.from_pretrained(
    model_id,
)  # 加载基础模型

sft_config = SFTConfig(
    output_dir='./sft_outputs',  # 设置输出目录
    report_to='swanlab',  # 设置报告工具
    num_train_epochs=EPOCHS,  # 设置训练轮数
    per_device_train_batch_size=1,  # 设置每设备训练批次大小
    learning_rate=LR  # 设置学习率
)


def find_all_linear_names(model):
    """查找模型中所有全连接层，用于 LoRA 目标"""
    cls = torch.nn.Linear  # 定义全连接层类
    lora_module_names = set()  # 创建一个集合来存储LoRA模块名称
    for name, module in model.named_modules():  # 遍历模型的所有模块
        if isinstance(module, cls):  # 检查模块是否为全连接层
            names = name.split('.')  # 将模块名称分割成列表
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  # 添加模块名称到集合
    if 'lm_head' in lora_module_names:  # 移除lm_head，因为通常不需要对其应用LoRA
        lora_module_names.remove('lm_head')
    return list(lora_module_names)  # 返回LoRA模块名称列表


target_modules = find_all_linear_names(model)  # 查找所有全连接层
print(f"target_modules\n{target_modules}")  # 打印目标模块列表
lora_config = LoraConfig(
    r=32,  # LoRA rank
    target_modules=target_modules,  # LoRA目标模块
    lora_alpha=64,  # LoRA alpha
    lora_dropout=0.1,  # LoRA dropout
    bias="none",  # LoRA bias
    task_type="CAUSAL_LM"  # 任务类型
)

trainer = SFTTrainer(
    model=model,  # 基础模型
    train_dataset=dataset,  # 训练数据集
    args=sft_config,  # SFT配置
    peft_config=lora_config,  # LoRA配置
)

trainer.train()  # 开始训练

trainer.save_model(final_adapter_path)  # 保存LoRA适配器

# 删除模型
del model

# 重新加载模型
model = AutoModelForCausalLM.from_pretrained(model_id)  # 加载基础模型

# 加载训练好的LoRA适配器
peft_model = PeftModel.from_pretrained(model, final_adapter_path)  # 从保存的路径加载LoRA适配器

# 合并LoRA权重到基础模型并卸载Peft包装
merged_model = peft_model.merge_and_unload()  # 合并LoRA权重并卸载LoRA适配器

# 保存合并后的完整模型
print(f"正在保存合并后的完整模型至: {final_merged_path}")  # 打印保存路径
merged_model.save_pretrained(final_merged_path)  # 保存合并后的模型
tokenizer.save_pretrained(final_merged_path)  # 保存tokenizer
print("合并后的模型保存完成")  # 打印完成信息
