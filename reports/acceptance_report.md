# 长文档切块检索验收报告

## 1. 切块边界

- `cjk-manual`: 485 字符 → 8 块，句中切点 0 处
- `ascii-manual`: 1644 字符 → 30 块，句中切点 0 处

样例（`cjk-manual` 第 0 块）：

- 字符区间 `[0, 82)`，结束边界 `sentence`
- 结尾 12 字符：`给出物流时效的参考区间。`
- 文本：客服知识库第一节说明退换货流程。第二节列出保修期的具体参数A，默认值为十。第三节说明发票申请的所需材料。第四节解释会员积分的计算方式。第五节给出物流时效的参考区间。

## 2. 同文档去重

- 去重前（块级 top-10）：['faq-03', 'product-manual', 'product-manual', 'product-manual', 'product-manual', 'product-manual', 'product-manual', 'product-manual', 'product-manual', 'product-manual']
- 命中不同文档数：2，重复命中 8 次
- 去重后（文档级 top-10）：['faq-03', 'product-manual', 'faq-04', 'faq-10', 'faq-01', 'faq-05', 'faq-13', 'faq-07', 'faq-08', 'faq-11']
- 重复命中：0 次

## 3. 聚合方式对比

- `max`: mentions-many(0.902641), mentions-once(0.865455), unrelated(0.595377)
- `mean_top_n`: mentions-many(0.902641), mentions-once(0.618384), unrelated(0.592913)
- `hit_count_weighted`: mentions-many(1.0), mentions-once(0.892364), unrelated(0.595377)

## 4. 增量更新

- 只更新 `doc-05`
- 未变动文档 19 篇，向量逐位相同：True
- 被更新文档向量是否变化：True

## 5. 关闭开关后的原有行为

- 开启切块：16 块；关闭切块：3 块
- 关闭后存储的向量与 `model.encode` 逐位一致：True
- 与 `semantic_search` 整篇编码结果一致：True

## 6. 千篇量级耗时

- 文档数 1000，块数 11000（每篇 11.0 块）
- 基线整篇编码：0.841s
- 切块建索引：2.056s（基线 2.444×）
- 无变更重建索引：0.001s，重新编码文档数 0
- 查询耗时：中位 0.01687s，P95 0.01947s

