# 论文 Method / Appendix 与当前实现的对齐建议

## 建议暂不直接覆盖原 TeX

本轮没有训练后结果，当前最重要的是得到可归因实验。建议先保留原稿，同时在实验成功后
选择下面两条路线之一。

### 路线 A：论文对齐当前可运行实现

- 将 `graph-conditioned lattice-MPC` 改为 `route- and conflict-conditioned typed primitive bank`；
- 将 OBS/NEU/PRIO 的 Transformer-GMM/diffusion 描述改为 source-typed analytic option basis
  with bounded dynamics residual；
- 将 online non-ego protocol 明确写成 logged replay；
- 将 `LOGGED_EGO`、`NEUTRAL_EGO` 定义为 auxiliary anchors，不属于 deployable macro intent；
- 将固定 macro set 改写为组合式 descriptor：
  `topology relation × negotiation order × longitudinal regime × continuous terminal condition`。

这条路线实现一致、风险较低，但需要依靠 same-root transport、witness certificate、option
preservation 和 dynamics-consistent natural options 来支撑 novelty。

### 路线 B：补齐论文原设定

- 实现真正的 learned observational multimodal decoder；
- 实现 ego-neutral intervention-conditioned decoder/diffusion；
- 实现 learned/rule reactive non-ego Waymax protocol；
- 实现图条件 OCP/lattice candidate optimizer；
- 对 macro coverage 做 topology-conditioned completeness audit。

这条路线潜在 novelty 更强，但工程和实验成本明显更高。只有 v16.1 main/ablation 不能达到
目标时，才建议投入该路线。
