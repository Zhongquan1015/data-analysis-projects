## Instacart Market Basket Analysis

This project explores association rule mining on transaction-level data from an online grocery platform, aiming to uncover frequently co-purchased products and higher-level purchasing patterns.

Using the transaction table as the core dataset, I applied both **Apriori** and **FP-Growth** algorithms to mine association rules under different support and confidence thresholds, and compared their efficiency in terms of runtime. For the strongest rules, multiple interestingness measures were computed, including **support, confidence, lift, and φ-coefficient**, to evaluate both statistical strength and practical relevance.

Beyond item-level analysis, products were further generalized into **departments** and **aisles**, enabling the discovery of more interpretable and stable association patterns at higher semantic levels. Key findings were visualized using network-style graphs, where nodes represent products or categories and edges represent strong or interesting associations.
