# 光刻设备产能建议（PyTorch 轻量化原型）

输入：周频市场/链条特征序列（默认回看 52 周）

输出：
- 建议目标库存（周）
- 建议产能调整比例（-max_ratio ~ +max_ratio）
- 置信度（0~1，来自模型学习到的不确定性）

## 快速开始（模拟数据）

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

训练并生成一个示例模型：

```bash
python -m litho_capacity_ai.train --data simulated --out_dir artifacts
```

用最新一期特征做推理：

```bash
python -m litho_capacity_ai.infer --model_dir artifacts --data simulated
```

## 快速开始（公开数据：全球+美国）

训练（会联网拉取 FRED + Stooq 的公开数据并缓存到 `data_cache/`）：

```bash
python -m litho_capacity_ai.train --data public --out_dir artifacts_public --start 2005-01-01
```

如果你只想要“快速可用”的稳定基线（只用 FRED 的核心序列，减少外网不稳定因素）：

```bash
python -m litho_capacity_ai.train --data public --public_profile fred_fast --out_dir artifacts_public --start 2005-01-01
```

推理（同样会拉取最新数据，并输出 asof_week）：

```bash
python -m litho_capacity_ai.infer --model_dir artifacts_public --data public
```

如果部署环境无法直连外网，可以先把公开数据下载为 CSV（每个序列一个文件，命名为 `SP500.csv`、`SOXX.csv` 等），然后用离线目录运行：

```bash
python -m litho_capacity_ai.train --data public --out_dir artifacts_public --start 2005-01-01 --offline_dir path/to/csv_dir
python -m litho_capacity_ai.infer --model_dir artifacts_public --data public --offline_dir path/to/csv_dir
```

## 接入真实数据

你只需要实现一个数据提供器，把实时数据整理成周频特征矩阵：
- shape: [lookback_weeks, num_features]
- 特征建议：价格/价差、波动率、订单/交期 proxy、宏观利率/汇率、上游材料价格等

然后在 `litho_capacity_ai/data` 下新增一个 provider，并在 `train.py` / `infer.py` 里通过参数选择即可。
