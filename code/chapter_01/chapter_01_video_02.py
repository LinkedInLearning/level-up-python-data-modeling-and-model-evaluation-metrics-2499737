from math import sqrt

TN = 2000
FP = 20
FN = 60
TP = 30

sens = TP / (TP + FN)

spec = TN / (TN + FP)

(sens + spec) / 2

TP / (TP + FP)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1 = 2*((precision*recall) / (precision+recall))

beta = 2

f_beta = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))

mcc = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

f1, f_beta, mcc = print()
