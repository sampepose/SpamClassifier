# spam-classifier
Classify spambase dataset: https://archive.ics.uci.edu/ml/datasets/Spambase

Statistics are based on a 70/30 training set split averaged for 50 runs.
## sklearn
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features (y=ham) | Top 5 Features (y=spam) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian | 0.808000 | 0.00781 | 0.84982 | 0.00714 | [('650', 1.2760192697768751), ('credit', 1.2476267748478689), ('hpl', 0.88242393509127726), ('people', 0.52748478701825663), ('font', 0.4292748478701825)] | [('credit', 2.2555208333333345), ('font', 1.3963862179487192), ('people', 0.54309294871794844), ('business', 0.53835737179487231), ('over', 0.50399038461538403)] |
| Multinomial | 0.87230 | 0.00714 | 0.95302 | 0.00390 | [('650', -1.9278529931605961), ('credit', -1.9801256215148824), ('hpl', -2.3107274101918582), ('people', -2.868116591218965), ('edu', -3.0283909589368152)] | [('credit', -1.4507395482450338), ('font', -1.9440271739962247), ('people', -2.8479883095744416), ('business', -2.90044359646099), ('over', -2.9511096702487905)] |
| Bernoulli (alpha=1.0, bin=0.31)| 0.89164 | 0.00528 | 0.95017 | 0.00403 | [('credit', -0.62493334953781776), ('people', -1.0074197209180786), ('hpl', -1.0558622403769), ('font', -1.2449217749113739), ('george', -1.3607226493637654)] | [('credit', -0.12910183231238115), ('font', -0.24123449696324872), ('people', -0.55563963955578277), ('over', -0.72801086549656979), ('3d', -0.73629591703067643)] |

#### Decision Trees
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features |
| --- | --- | --- | --- | --- | --- |
| DecisionTreeClassifier(criterion="entropy") | 0.91136 | 0.00758 | 0.91214 | 0.00853 | [('remove', 0.21730246077008433), ('free', 0.1383136780699713), ('hp', 0.078559911850530045), ('money', 0.066369886207552659), ('george', 0.054680657133664989)] |

#### Random Forest
// TODO

[('free', 0.098474220547133465), ('remove', 0.092541805565311538), ('your', 0.09022347711132328), ('you', 0.061481529787477354), ('000', 0.061045877750548337)]
Accuracy. Avg: 0.93936, Std: 0.00656
AUC. Avg: 0.97864, Std: 0.00362
[Finished in 34.691s]

## My Implementation
#### Naive Bayes
| Method | Accuracy Avg | Accuracy Std | AUC Avg | AUC Std | Top 5 Features (y=ham) | Top 5 Features (y=spam) |
| --- | --- | --- | --- | --- | --- | --- |
| Gaussian | 0.80858 | 0.01097 | 0.85702 | 0.00859 | [('650', 1.3194578005115096), ('credit', 1.2579437340153434), ('hpl', 0.94403580562659861), ('people', 0.53731969309462912), ('george', 0.42955498721227592)] | [('credit', 2.2747826086956491), ('font', 1.3878418972332005), ('business', 0.54054545454545522), ('people', 0.5380316205533594), ('over', 0.51667193675889345)] |
| Multinomial | 0.86884 | 0.00866 | 0.95108 | 0.00528 | [('credit', 0.14393208823250517), ('650', 0.14351229786150893), ('hpl', 0.09480940574430724), ('people', 0.056671145539713377), ('font', 0.048257592687994753)] | [('credit', 0.23500355935145184), ('font', 0.14540852047528446), ('people', 0.056419703295146083), ('over', 0.052507351381355288), ('business', 0.051720856290241896)] |
| Bernoulli (alpha=1.0, bin=0.31) | 0.87806 | 0.00736 | 0.95719 | 0.00401 | [('credit', 0.10577409242592786), ('people', 0.072504803316816663), ('hpl', 0.069268884619273968), ('font', 0.057033067044190547), ('george', 0.050156739811912252)] | [('credit', 0.1190450352685837), ('font', 0.1069994574064025), ('people', 0.077590884427563678), ('3d', 0.066087900162778032), ('over', 0.065653825284861606)] |


## Analysis
### Naive Bayes
Three methods of Naive Bayes classifiers were tested: Gaussian distribution, multinomial, and multi-variate Bernoulli. The Gaussian NB assumes that the values of each feature are continuous and distributed normally. In multinomial NB, a document d is modeled as the outcome of |d| independent trials from the vocabulary. Typically, a document is represented as a vector of word counts or word frequencies. The multi-variate Bernoulli NB represents a document as a binary vector over the space of the vocabulary. Each document can be seen as a collection of multiple independent Bernoulli experiments, one for each word in the vocabulary [1].
<br /><br />

Table 1. Accuracy and AUC for Naive Bayes Methods

| Method | Accuracy | AUC |
| --- | --- | --- |
| Gaussian | 80.800% +/- 0.216% | 84.982% +/- 0.198%
| Multinomial | 87.230% +/- 0.198% | 95.302% +/- 0.108%
| Bernoulli | 89.164% +/- 0.146% | 89.164% +/- 0.112%

Table 1 summarizes metrics for each method. Based on accuracy, Bernoulli appears to be the better classifier, however multinomial beats it out based on AUC. This is to be taken with a grain of salt, as a study [2] does not believe "standard auc is a good measure for spam filters, because it is dominated by non-high specificity (ham recall) regions, which are of no interest in practice."
<br /><br />

Table 2. Multinomial model training with frequency vs binary word occurrence vectors

| Method | Accuracy | AUC |
| --- | --- | --- |
| Frequency | 87.230% +/- 0.198% | 95.302% +/- 0.108% |
| Binary | 87.954% +/- 0.230% | 95.756% +/- 0.136% |

Previous research [3] inspired a multinomial model to be trained using binary word occurrence vectors instead of frequency vectors. The results in Table 2 show a slight increase in both accuracy and AUC when using a binary word occurrence vector instead of the usual word frequency vector, which is consistent with the findings of [3].

Although [2] demonstrates that the binary multinomial model should yield better results than the Bernoulli model, this did not occur with the given data. This is because the vocabulary is not large enough, as shown from the accuracy results in [3]. I suspect that an increase in vocabulary size would show the multinomial model surpasses the Bernoulli model.  

### References
[1] A.  McCallum and K.  Nigam, "A comparison of event models for naive bayes text classification", AAAI-98 workshop on learning for text categorization, vol. 752, pp. 41-48, 1998.

[2] V.  Metsis, I.  Androutsopoulos and G.  Paliouras, "Spam Filtering with Naive Bayes – Which Naive Bayes?", in Conference on Email and Anti-Spam, Mountain View, California USA, 2006.

[3] K.  Schneider, "On word frequency information and negative evidence in Naive Bayes text classification", EsTAL, vol. 3230, pp. 474-486, 2004.
