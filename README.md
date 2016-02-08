# spam-classifier
Classify spambase dataset: https://archive.ics.uci.edu/ml/datasets/Spambase

Statistics are based on a 70/30 training set split averaged for 50 runs.
## sklearn
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features (y=spam) | Top 5 Features (y=ham) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian | 0.808000 | 0.00781 | 0.84982 | 0.00714 | [('650', 1.2760192697768751), ('credit', 1.2476267748478689), ('hpl', 0.88242393509127726), ('people', 0.52748478701825663), ('font', 0.4292748478701825)] | [('credit', 2.2555208333333345), ('font', 1.3963862179487192), ('people', 0.54309294871794844), ('business', 0.53835737179487231), ('over', 0.50399038461538403)] |
| Multinomial | 0.87230 | 0.00714 | 0.95302 | 0.00390 | [('650', -1.9278529931605961), ('credit', -1.9801256215148824), ('hpl', -2.3107274101918582), ('people', -2.868116591218965), ('edu', -3.0283909589368152)] | [('credit', -1.4507395482450338), ('font', -1.9440271739962247), ('people', -2.8479883095744416), ('business', -2.90044359646099), ('over', -2.9511096702487905)] |
| Bernoulli | 0.89164 | 0.00528 | 0.95017 | 0.00403 | [('credit', -0.62493334953781776), ('people', -1.0074197209180786), ('hpl', -1.0558622403769), ('font', -1.2449217749113739), ('george', -1.3607226493637654)] | [('credit', -0.12910183231238115), ('font', -0.24123449696324872), ('people', -0.55563963955578277), ('over', -0.72801086549656979), ('3d', -0.73629591703067643)] |

#### Decision Trees
// TODO

#### Random Forest
// TODO

## My Implementation
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features |
| --- | --- | --- | --- | --- | --- |
| Gaussian | 0.80858 | 0.01097 | 0.85702 | 0.00859 | [('650', 1.3194578005115096), ('credit', 1.2579437340153434), ('hpl', 0.94403580562659861), ('people', 0.53731969309462912), ('george', 0.42955498721227592)] | [('credit', 2.2747826086956491), ('font', 1.3878418972332005), ('business', 0.54054545454545522), ('people', 0.5380316205533594), ('over', 0.51667193675889345)] |
| Multinomial | x | x | x | x | x |
| Bernoulli | x | x | x | x | x |