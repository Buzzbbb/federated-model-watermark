# 联邦学习模型水印验证框架

`federated-model-watermark` 是一个信息隐藏与网络空间安全方向的可运行开源项目，包含核心算法代码、命令行入口、实验配置、示例脚本和 smoke tests。

## Overview

本项目面向联邦学习场景中的模型版权保护，提供参与方模拟、模型训练、触发样本构造、水印嵌入、聚合后验证和鲁棒性评估等模块。框架可比较不同水印策略在非独立同分布数据、客户端掉线和模型剪枝条件下的表现，并输出准确率、水印成功率和误触发率等指标。项目适用于隐私计算、模型知识产权保护和人工智能安全课程实验，也方便替换不同模型结构、聚合策略、实验数据和触发样本配置方案。

## Features

- 统一的数据加载、实验配置和结果保存流程
- 面向信息隐藏/数字水印/隐写分析任务的模块化设计
- 支持实验指标输出、样例结果归档和后续算法扩展
- 适合课程实验、毕业设计、论文复现实验和课题组日常开发

## Quick Start

```bash
python examples/demo.py
python -m unittest discover -s tests
python -m federated_model_watermark.cli --message "demo payload" --report docs/cli_report.md
```

## Keywords

federated learning · model watermark · ownership verification · AI security

## Authors

- 负责人：林裕斌
- 参与人：曾科、田承金
- 指导教师：吕善翔
- 单位：暨南大学网络空间安全学院

## License

本项目采用 MIT License 开源。Copyright (c) 2026 Lin Yubin, Zeng Ke, Tian Chengjin, Shanxiang Lv, Jinan University.
