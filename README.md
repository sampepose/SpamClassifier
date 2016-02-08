# spam-classifier
Classify spambase dataset: https://archive.ics.uci.edu/ml/datasets/Spambase

Statistics are based on a 70/30 training set split averaged for 50 runs.
## sklearn
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features |
| --- | --- | --- | --- | --- | --- |
| Gaussian | 0.808000 | 0.00781 | 0.84982 | 0.00714 | 'meeting' >>> 'data', '85', 'labs', 'conference' |
| Multinomial | 0.87230 | 0.00714 | 0.95302 | 0.00390 | 'meeting', 'data', '857', 'labs', 'conference' |
| Bernoulli | 0.89164 | 0.00528 | 0.95017 | 0.00403 | 'labs' = 'meeting', 'data', '650' = '85' |

#### Decision Trees
// TODO

#### Random Forest
// TODO

## My Implementation
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features |
| --- | --- | --- | --- | --- | --- |
| Gaussian | 0.80858 | 0.01097 | 0.85702 | 0.00859 | ('650', 1.311463539010707), ('credit', 1.272957674655788), ('hpl', 0.90324324324324312), ('people', 0.53365119836817931), ('font', 0.43681285058643571) |
| Multinomial | x | x | x | x | x |
| Bernoulli | x | x | x | x | x |