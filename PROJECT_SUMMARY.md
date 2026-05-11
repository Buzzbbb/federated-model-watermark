# 联邦学习模型水印验证框架

英文名称：`federated-model-watermark`

开源地址：`https://github.com/Buzzbbb/federated-model-watermark`

项目时间：2025年7月-至今

## 作者信息

- 负责人：林裕斌，专业：网络空间安全，硕士生
- 参与人：曾科，专业：网络空间安全，硕士生
- 参与人：田承金，专业：网络空间安全，硕士生
- 指导教师：吕善翔，网络空间安全学院教师

## 项目内容

本项目面向联邦学习场景中的模型版权保护，提供参与方模拟、模型训练、触发样本构造、水印嵌入、聚合后验证和鲁棒性评估等模块。框架可比较不同水印策略在非独立同分布数据、客户端掉线和模型剪枝条件下的表现，并输出准确率、水印成功率和误触发率等指标。项目适用于隐私计算、模型知识产权保护和人工智能安全课程实验，也方便替换不同模型结构、聚合策略、实验数据和触发样本配置方案。

## 影响力

项目可为联邦学习安全与模型水印研究提供基础框架，帮助学生理解分布式训练环境下模型所有权验证与鲁棒性评估问题。

## 开发语言

Python

## 代码规模

1012行（按当前项目 src/tests/examples 下 Python 代码统计）

## 建议仓库结构

```text
federated-model-watermark/
├── README.md
├── LICENSE
├── PROJECT_SUMMARY.md
├── src/
├── examples/
├── tests/
├── docs/
└── screenshots/
```

## 截图材料

- 项目目录截图：`screenshots/directory.png`
- 项目说明截图：`screenshots/readme.png`
- 项目声明截图：`screenshots/license.png`

## 关键词

federated learning, model watermark, ownership verification, AI security
