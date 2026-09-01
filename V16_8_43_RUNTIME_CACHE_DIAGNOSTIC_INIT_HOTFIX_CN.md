# V16.8.43 runtime hotfix：cache diagnostic 初始化遗漏

## 结论

本修复是**纯工程可靠性修复，不是算法修改**。它修复 `profile8_parallel2` 首次
进入 interaction hypothesis 后出现的：

```text
KeyError: 'interaction_environment_compatibility_cache_hits'
```

原 V43 在 `_interaction_aware_recovery_certificate_np` 内部已经初始化三个 cache-hit
字段，但外层 `_construct_interaction_aware_reachable_response_envelope_np` 在聚合这些
字段时没有先初始化同名 key，因此第一次执行 `detail[key] += value` 就会崩溃。

同一遗漏还包含两个潜在后续错误：

```text
interaction_joint_compatibility_cache_hits
interaction_successor_context_cache_hits
```

只修第一个报错字段会留下另外两个潜在崩溃路径，因此本 hotfix 一次性补全三个字段。

## 修改

仅在 outer constructor 的 diagnostic `detail.update({...})` 中加入：

```python
"interaction_environment_compatibility_cache_hits": 0,
"interaction_joint_compatibility_cache_hits": 0,
"interaction_successor_context_cache_hits": 0,
```

不改变：

- V39/V42/V43 hard certificate；
- blocker/root/response support；
- same-root burden constraints；
- environment / joint CSP；
- physical tube / shift closure；
- selector / fallback score；
- controller / action execution；
- p_min、root mass、beta、8 s horizon；
- counterfactual48/fresh37 promotion Gate。

因此旧实验指令完全兼容，不需要改命令。

## 为什么原测试漏掉

原 V42 constructor 测试重点覆盖 V39 nested early-return；V43 cache test 直接调用内层
certificate。二者都没有覆盖：

```text
real outer V42 constructor
→ interaction hypothesis
→ inner certificate
→ aggregate cache diagnostics
```

新增 regression 现在真实走这条聚合路径，并构造两个共享 first action 的 hypothesis，
同时覆盖 successor-context cache hit。

## 验证

- V43 dedicated tests：8/8 passed；
- V16.8.25→43 focused semantic/integrity sanity：104/104 passed；
- Python compile：passed；
- 5 个冻结 manifest hash：passed；
- 原 `profile8_parallel2` / `counterfactual48_parallel2` / analyzer 命令均未修改。

服务器上的 Waymax profile8 仍需重新执行，以验证端到端性能；本地环境没有用户的
Waymax dataset/checkpoint/runtime，因此没有伪造 profile8 结果。
