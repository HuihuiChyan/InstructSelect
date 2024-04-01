# Instruction Data Mining based on Representations

这些代码可以用来筛选更好的Instruction数据，用于指令微调大模型。

但是必须要说明的是，由于缺乏一个合理的评价方法，这些方法并未得到充分的验证。

在实际使用中，建议还是使用全部的指令微调数据，或使用手工设置的统计规则进行筛选。

### ⚡️ 依次执行以下文件，获取基于embedding聚类的数据筛选结果

* -create_openai_embeddings 获取数据的openai嵌入

* -dimension_reduction_pca 基于pca方法对embedding进行降维

* -create_cluster_visualization_dbscan 基于dbscan方法对embedding进行聚类

* -create_cluster_visualization_hdbscan 基于hdbscan方法对embedding进行聚类

* -create_cluster_visualization_kmeans 基于kmeans方法对embedding进行聚类

* -sample_densemax 基于embedding，筛选能够使密度最大的instruction子集

* -visualizer.html 可以借助该文件，在浏览器中可视化获取的instruction embedding

### ⚡️ 按照下面的步骤执行，完成数据筛选和验证的整个流程

1. 基于representations或者其他方法，筛选指令数据；

2. 使用开源的微调框架（比如FastChat），使用指令数据进行微调；

3. 仿照evaluate.sh，使用alpaca_eval和mt_bench验证训练结果；