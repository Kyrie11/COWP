# Cache 复用证据

- `cache_sufficiency_full_uploaded.json`：用户上传的原始 Waymax cache 完整扫描，记录 train=14640、val=5013，并给出 `REUSE_WITH_LOGDIV_DISABLED`。
- `v14_eval/cache_alignment_train.json`：上一轮 transport_v9 train 抽样对齐，记录 raw/overlay=20440/20440、pass=true。
- `v14_eval/cache_alignment_val.json`：上一轮 transport_v9 val 抽样对齐，记录 raw/overlay=5013/5013、pass=true。
- `v14_eval/model_anchor_preflight_val.json`：模型实际数据路径、critical mapping、anchor 和 typed basis 预检。
- `v14_eval/natural_oracle_val.json`：v9 natural label-space oracle 诊断。

由于两个报告的 train 文件数不同，服务器执行时必须以新的 live cache gate 为准。
