# ReactSim 归一化 A/B × SMART/MTR 结果接口

ReactSim 作者的官方程序输出格式尚未在用户的服务器上经验证。本模块**不尝试直接解包未知 pickle**；请先用作者源码中的真实 actor token、ego 历史、仿真输出，把以下字段逐项核对并写成 JSONL。没有对应数据，字段不要填写虚构值；该场景会被排除而非自动生成 evidence。

每个 `(scene_id, action_id, response_model)` 恰好一个记录：

```json
{"scene_id":"scene_0001","action_id":"merge_early","response_model":"SMART","initial_state_digest":"sha256_of_same_initial_state","affected_actor_pair_digest":"sha256_of_fixed_two_agent_tokens","action_trajectory_digest":"sha256_of_frozen_ego_trajectory","source_rollout_digest":"sha256_of_real_native_output","progress_m":52.5,"ego_collision":false,"third_party_collision":false,"hard_brake":false,"rule_valid":true,"mechanism_executed":true,"simulator_completed":true}
```

请注意：上面只是**格式例子**，不是执行结果，也没有任何真实模型的数字证明。必须将同场景的 `merge_early`/`merge_late` 在 SMART 和 MTR 两个环境中分别执行，得到 4 条记录。不同 response model 的同一 action 需用**完全相同** ego trajectory digest。`initial_state_digest` 和固定第三方 actor pair digest 四条必须一致；计算哈希时必须先标准化排序、时间索引和坐标精度。

```bash
python -m ride.reactsim_paired_audit \
  --input /path/to/verified_4arms.jsonl \
  --output /path/to/posthoc_two_model_audit.json \
  --action-a merge_early --action-b merge_late \
  --model-a SMART --model-b MTR --progress-tolerance-m 3
```

输出中的 `strict_posthoc_reversals` 是**反应环境造成的后验风险排序差异**，并非某个 planner 在动作前实际选错。要升级到正式 RQ1，必须另行提供 ex-ante 响应模型评分和锁定的 `selected_action` 及其时间/commit hash；原项目中 `ride.selectors`、`ride.decision` 已有前承诺读取器，但不是官方 GameFormer/RACP 的模型适配器。风险排序应同时报告第三方碰撞、持续减速度、短 TTC，并按场景 clustering。
