# 制造感知可微镜头设计

本项目在 DeepLens 上加入制造误差采样、相对成本目标和多轮材料搜索，
交替优化玻璃组合与曲率、厚度、非球面等光学参数。

## 安装与运行
建议 Python 3.12 及以上，使用独立环境。先按 PyTorch 官方安装指南安装
相互匹配的 torch/torchvision（CPU 或 CUDA），再运行：

```powershell
python -m pip install -r requirements.txt
python run.py --config configs/demo_search.yml --check
python run.py --config configs/demo_search.yml
```

`--check` 检查环境、配置、输入与材料；通过后再开始优化。
程序使用当前环境的 Python，不依赖作者电脑路径。进度监视为英文，包含
已用时间及在已有记录充足时的时间估计。日志保存在 results/runtime_logs。
Ctrl+C 会停止本次前台搜索，已完成阶段可以复用。

## 调整计算量
编辑 configs/demo_search.yml：
- max_rounds：搜索轮数；max_trials_per_round：每轮候选数。
- beam_width：同时保留的材料路径数。
- nominal_steps_per_trial：每个候选的名义恢复步数。
- robust_steps_per_trial：每个候选的制造感知精调步数。
- evaluation_trials：独立制造误差抽样数。
- device：auto 自动选择 CUDA/CPU，也可指定 cuda。

演示默认只有两轮，每个阶段三步，用于检查软件流程，不代表论文收敛结果。
多轮搜索可累积更换多片材料，并非整个训练只能替换一片。

## 演示与论文实验的区别
公开演示采用 Cooke 示例几何结构、虚构数学材料和假设成本，未包含供应商
价格表、真实投影镜头及工程玻璃库。材料名称 h-demo-* 并非真实玻璃牌号。
它可以展示程序流程，但不能复现论文中的成本和 RMS 改善百分比。
论文主实验及三组消融配置另存于 configs/paper/，需导入授权研究输入。
详见 docs/DATA.md；原理、术语及代码模块见 docs/METHOD.md。

已在作者现有 Python 3.14.6、PyTorch 2.14.0+cu126、RTX 3060 环境完成
两轮 CUDA 搜索，并检查论文主实验及完整方法配置的输入。尚未检查全新
安装环境，也未重新运行完整论文实验。小示例结果在 examples/，仅用于软件演示。

完成四组论文消融搜索后，可运行 `python tools/evaluate_paper_ablations.py`
进行共同误差样本评价与已保存 Q90 差异的自助重采样分析。

## 结果解释
相对制造成本指数反映给定假设下的投入；RMS p90 是误差样本 RMS 的90%分位值；
模拟通过率是指定误差模型与门槛下的抽样结果。三者均需与实际制造数据区分。
几何 MTF 估计用于辅助像质评价，不等同于实测 MTF。

## 开源与引用
保留 DeepLens 原作者署名与 Apache-2.0 许可证，新增方法见 NOTICE。
方法作者：Taiyuange Lou。GitHub 用户名：yogurt-lt。
拟发布仓库：https://github.com/yogurt-lt/manufacturing-aware-lens-design
目前尚未上传 GitHub；论文 DOI 在发表后补充。
发布步骤见 docs/PUBLISH_zh.md。
