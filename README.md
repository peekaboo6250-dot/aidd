# AIDD：肾纤维化虚拟筛选起步骨架

面向 RTX 3060 Laptop 6GB、16GB 内存，先验证离线小规模流程。
**当前没有训练好的深度学习模型，没有真实对接或药效预测，也未确定生物学靶点。**

## 安装与首轮运行

推荐 Linux / WSL2；先在 CPU 上完成以下步骤，不需要 CUDA。

```bash
git clone https://github.com/peekaboo6250-dot/aidd.git
cd aidd
conda env create -f environment.yml
conda activate aidd
python -m pytest -q
python -m aidd.cli benchmark --output outputs/pilot
```

不用 Conda 可用 Python 3.11/3.12 创建 venv，再执行 `python -m pip install -e '.[dev]'`。
直接依赖版本固定于 pyproject.toml；这不是跨平台完整传递依赖锁文件。
成功后保存 `python -m pip freeze > outputs/environment-resolved.txt`，配合报告记录环境。

## 目录与模块

| 路径 | 内容 |
|---|---|
| `configs/pilot.json` | CPU、batch=1、workers=0、seed=42、默认上限1000 |
| `data/examples/` | 手工编写的8条化合物和3条虚构靶点评分，无实验标签 |
| `data/raw/` | 原始数据，默认不入 Git |
| `data/processed/` | 后续清洗数据，默认不入 Git |
| `src/aidd/targets.py` | 人工证据表加权排序 |
| `src/aidd/compounds.py` | RDKit解析、标准化、取母体、保留立体信息、去重、拒绝日志 |
| `src/aidd/interfaces.py` | AI对接/GNN/不确定性协议与占位类 |
| `models/` | 将来存放权重；不提交大文件 |
| `outputs/` | CSV、JSON报告，默认不入 Git |
| `tests/` | 数据边界、重现性、接口与端到端测试 |
| `docs/` | 靶点证据、模型接入和基准说明 |

## 首轮基准验收

8条输入 → 5条有效唯一母体、3条拒绝（重复结构、非法SMILES、空SMILES）；3条示例靶点。
输出 compounds.csv、rejected.csv、targets.json、predictions.json 和 report.json。
报告记录输入 SHA256、配置、软件版本、平台、耗时。相同输入的清洗和预测文件应一致；
运行时间与平台字段允许变化。seed 预留给未来随机模型；当前流程没有随机操作。
模拟 GNN 恒为0分，仅用于检查数据传递；不应据此排序或挑选候选药物。

```bash
python -m aidd.cli benchmark --output outputs/run1
python -m aidd.cli benchmark --output outputs/run2
python -m pytest -q
```

默认超过1000条就报错，不静默截断；绝对试运行上限10000条。没有批量下载或大规模筛选入口。
首版不生成3D构象、不枚举质子化/互变异构体；取母体会去掉盐和部分混合物信息，原始SMILES需保留。
可用元素与100重原子限制是工程保护条件，不是药物相似性或疗效筛选标准。

## 本机资源策略

本版实际使用 CPU，显存占用为0（不计其他应用），逐行读取并限制样本量。
`gpu_memory_budget_gb=4` 是未来适配器的设计预算，当前不会调用CUDA或实施显存限额。
后续GPU适配器先采用batch=1、workers=0、单模型串行推理，独立环境安装模型官方依赖；
逐次记录峰值显存，OOM时减小图/构象数量，无法满足时退回CPU或换模型。
不承诺所有AI对接模型都能装进6GB，不建议先安装多个相互冲突的CUDA/GNN依赖。

## 开发范围

见 [后续模型和靶点工作](docs/next_steps.md)。本仓库只建立可复现工程起点，不能作为疾病证据或方法学创新结论。
官方参考：
- [RDKit Python入门](https://www.rdkit.org/docs/GettingStartedInPython.html)
- [MolStandardize接口](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html)
- [PyTorch CUDA内存说明](https://docs.pytorch.org/docs/stable/notes/cuda.html)
