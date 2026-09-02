# OC-RAP 论文、数据、代码与 v48.34 结果审计

## 1. 论文核心 idea

论文的中心论点不是“碰撞概率更低”，而是把 **deployable recoverability** 作为规划动作的一级准入与优化量。

对每个可执行候选前缀，算法依次处理：

1. 预测 recovery-sufficient latent roots；
2. 估计执行前缀之后可获得的 observation embedding；
3. 用 observation-equivalence kernel 把部署时仍不可区分的 latent roots 合并；
4. 在每个等价类内只允许选择一个共享 recovery option；
5. 用 affordance-conditioned recovery margin 和 lower-tail aggregation 得到 deployable recoverability；
6. 同时计算 oracle recoverability，并形成 oracle-to-deployable gap（ODG）；
7. 通过 OC-MERO/CRISP 进行校准准入：nominal 仍有足够 deployable headroom 时尽量不干预，只有 nominal 消耗恢复空间或依赖 oracle artifact 时才选择保护性候选。

最关键的运算顺序是：

- Oracle：每个隐藏 root 可先各自选择最优 recovery option；
- Deployable：必须先按 post-prefix observation 合并不可区分 roots，再在每个 class 内选一个共享 option。

因此论文真正针对的是 **false recoverability admission**：某个动作看起来每条隐藏未来都有恢复办法，但部署车辆不知道自己处于哪条隐藏未来，实际上无法选到正确恢复动作。

主要论文指标：

- FRA：false recoverability admission，越低越好；
- ODG：oracle-to-deployable gap，越低越好；
- DRS：deployable recovery success，越高越好；
- NUP：nominal utility preservation，越高越好；
- 以及碰撞、offroad、intervention、post-contact recovery 与 calibration 指标。

## 2. 数据集审计

上传的 reports 表明数据合同完整覆盖 safe、near-contact、contact 的 train/val/test/calibration：

| Regime/Split | Samples | Scene-time groups | Scenes | 平均 candidates/group |
|---|---:|---:|---:|---:|
| Safe train | 20,000 | 2,500 | 1,171 | 8.00 |
| Safe val | 2,328 | 291 | 132 | 8.00 |
| Safe test | 3,216 | 402 | 175 | 8.00 |
| Safe calibration | 2,544 | 318 | 135 | 8.00 |
| Near train | 13,324 | 1,800 | 600 | 7.40 |
| Near val | 3,445 | 433 | 176 | 7.96 |
| Near test | 4,723 | 595 | 250 | 7.94 |
| Near calibration | 6,039 | 765 | 316 | 7.89 |
| Contact train | 16,790 | 2,000 | 500 | 8.39 |
| Contact val | 6,477 | 723 | 211 | 8.96 |
| Contact test | 6,687 | 747 | 209 | 8.95 |
| Contact calibration | 16,843 | 1,896 | 543 | 8.88 |

每个 scene-time group 恰有一个 nominal candidate。闭环主表默认以独立 scene 为统计单位，每 scene 取一个确定性 target；这会分别覆盖 Safe 175、Near 250、Contact 209 个独立 test scenes。离线评测仍使用全部 candidate groups。

## 3. 代码与论文的对应关系

代码已经包含论文思想的主要计算对象，但实验迭代后并非逐公式一一对应：

- candidate prefix、semantic macro、utility、feasibility：对应论文可执行候选动作；
- latent roots、root probability、recovery options、`m_star`：对应 recovery margin tensor；
- oracle 方向：先 option-max，再 root aggregation；
- deployable 方向：observation consistency、shared option、lower-tail/robust aggregation；
- direct recovery/evidence heads、eligible set policy、barrier-gated slack：是 v48 系列中对 OC-MERO/CRISP 的工程化实现；
- closed-loop runner 在执行选中前缀后重新规划，并输出 FRA/DRS/ODG/NUP 与 Waymax 几何指标。

当前实现最重要的协议是：deployable selector 不应读取 teacher margin、hard violation 或 harm proxy；teacher 只可用于选中动作后的 audit。外部 baseline 也遵循同一原则，只有明确命名的 oracle upper bound 可以使用 teacher tensor。

## 4. v48.34 RC=20 结果解释

上传结果显示：

- pipeline 本身有效；
- certificate controller 已执行；
- Natural gate 未通过，最终退出码 20；
- held-out test roots 尚未读取；
- 因此没有授权正式 test/stress 或论文主结论。

失败主层是 `development_rule_fit`，而不是数据、训练或模型加载的工程失败。

### Balanced variant / Near-contact

- candidate-level positive AUC：约 0.8702；
- legacy evidence-only positive AUC：约 0.8434；
- proposal oracle feasible：是，找到 9 个 safe-positive opportunity groups；
- certificate verify：290 groups 中只选 5 个，其中 1 个 positive；precision=0.20，positive recall=0.1111；
- 选中的 5 个中没有 harmful selected，但选择覆盖率太低，且正向区分不足。

解释：**候选生成/候选层信号是有用的，但准入与最终排序没有把这些机会稳定转化为部署动作。** Near-contact 最值得观察的闭环物理指标是 minimum clearance、TTC、critical-TTC exposure、terminal clearance/TTC，以及是否以较小 NUP 损失换取更大恢复空间。

### Balanced variant / Contact

- candidate-level positive AUC：约 0.5570，接近弱区分；
- proposal oracle feasible：是，找到 20 个 safe-positive opportunity groups；
- certificate verify：764 groups 中选 41 个，positive 为 0，harmful 为 16；
- learned selector 在 contact 上没有把 oracle proposal support 转化为可靠选择。

解释：Contact 的 candidate set 中存在潜在恢复候选，但当前 learned admission/ranking 明显不可靠。不能从 RC=20 certificate 声称 contact 优势。闭环视频只能按真实物理结果筛选：terminal clearance、free-space AUC、clearance gain、escape、stable stop、re-contact 和 offroad。

### Precision variant

Precision 在 near 的 development fit 上选得更多，但 certificate verify 中 9 个 selected 只有 1 个 positive、4 个 harmful；因此不能仅凭“更积极干预”把 precision 当作主模型。Balanced/precision 应作为预先声明的独立诊断 variant 运行，不能看 held-out 结果后再选择赢家。

## 5. 本次 external-baseline 实现对主论文比较的影响

原 baseline 代码已经不是纯占位符，但存在四类会影响结论的问题：

1. MARC/RACP/PSF 只保留了有限候选格代理，缺少明确的 non-anticipative temporal branch risk；
2. `min_ttc` 语义错误，记录的是最近距离时刻；
3. 同一 candidate group 中每个方法、每个候选重复预测周车未来；
4. 闭环脚本没有强制绑定 regime target set，可能比较不同场景。

v49 已修正这些问题，并对每个 baseline 生成明确 fidelity 审计。对于 planning 代码未公开或依赖连续动力学优化的论文，本实现保持统一可执行 candidate protocol 下的 paper-core adaptation，不冒充作者官方实现。

## 6. 视频结论边界

视频筛选不会预设“OC-RAP 一定更好”。它只在完成配对闭环后，从实际指标 delta 中选出：

- Near：TTC/clearance/critical exposure 改善，且不增加 collision/offroad；
- Contact：terminal clearance、free-space AUC、escape/stable-stop 改善，且不增加 re-contact/offroad。

如果严格条件下不足 5+5 条，脚本会失败，而不是降低标准凑够 10 条。由于上传包没有 checkpoint，本次交付不包含伪造的数值表或视频；实际结果必须在原训练服务器运行后生成。
