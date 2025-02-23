# Maorkov

### Fake Code

```
1. Check the success status.
if target_gene_frequency >= 1:
    Success
if generation > max_generation:
    Failure
if total_budget > max_budget:
    Failure
2. Recursive cross
for father_num in father_num_list:
    pop = self_cross(pop, father_num)
    target_gene_frequecy = pop.target_gene_frequecy
    generation += 1
    total_budget += father_num
    goto 1
```

### 初始参数



### 初始群体

初始群体一: 背景基因按照基因频率设置, 目标基因编辑为全 0

初始群体二: 背景基因按照基因频率设置, 目标基因编辑为全 1

上述两个群体交配得到 F1

### F1

群体内交配, 按照目标基因携带数量进行同质选配

交配若干代, 统计基因型

基因型达到目标停止交配, 得到 Fn

### Fn




