# Evaluation of Segmentations


## Semantic Evaluation:
In the semantic evaluation all pixels are considered equal.
Values are calculated in a **case-wise** manner and are **then aggregated** over the whole dataset.
## Cases (considered per class):
All values are pixel-wise
TBD

## Instance Evaluation:
Through connected components all pixels are grouped into instances.
For **each case the instance-wise values are calculated, then averaged over the case and then over the whole dataset**.
## Cases (considered per class):
All values are pixel-wise
1. No GT Segmentation in Image and No PD
Precision = 
Recall = 
DICE =
2. No GT Segmentation in Image but PD
Precision = 
Recall = 
DICE = 
3. GT Segmentation in Image but No PD
Precision = ?
Recall = 0
DICE = ?
4. GT Segmentation in Image and PD
Precision = (TP / (TP + FP))
Recall = (TP / (TP + FN))
DICE = (2 * TP) / (2 * TP + FP + FN)
