TN = 2000
FP = 20
FN = 60
TP = 30

accuracy = (TN + TP) / (TN + FP + FN + TP)

accuracy


sensitivity = TP / (TP + FN)

specificity = TN / (TN + FP)

balanced_accuracy = (sensitivity + specificity) / 2

balanced_accuracy
