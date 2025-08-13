# overall

| model | accuracy |
|-------|----------|
| resnet | 0.5537 |
| cnn1d | 0.5494 |
| cnn2d | 0.4731 |
| transformer | 0.4786 |
| resnet + augment | 0.5934 |


# v1
错误噪声标准差计算
length_scale=1
# v2
修正噪声标准差计算
length_scale=50
# v3
修正噪声标准差计算
length_scale=1
# v4
错误噪声标准差计算
length_scale=50

# resnet

Overall accuracy: 0.5537
SNR -20.0 dB: Accuracy = 0.0924
SNR -18.0 dB: Accuracy = 0.1027
SNR -16.0 dB: Accuracy = 0.0887
SNR -14.0 dB: Accuracy = 0.1244
SNR -12.0 dB: Accuracy = 0.1560
SNR -10.0 dB: Accuracy = 0.1817
SNR -8.0 dB: Accuracy = 0.3182
SNR -6.0 dB: Accuracy = 0.4110
SNR -4.0 dB: Accuracy = 0.5698
SNR -2.0 dB: Accuracy = 0.7006
SNR 0.0 dB: Accuracy = 0.7702
SNR 2.0 dB: Accuracy = 0.8174
SNR 4.0 dB: Accuracy = 0.8241
SNR 6.0 dB: Accuracy = 0.8379
SNR 8.0 dB: Accuracy = 0.8543
SNR 10.0 dB: Accuracy = 0.8514
SNR 12.0 dB: Accuracy = 0.8452
SNR 14.0 dB: Accuracy = 0.8464
SNR 16.0 dB: Accuracy = 0.8416
SNR 18.0 dB: Accuracy = 0.8179

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.54      0.57      0.55      4000
      AM-DSB       0.52      0.53      0.52      4000
      AM-SSB       0.28      0.24      0.26      4000
        BPSK       0.58      0.62      0.60      4000
       CPFSK       0.60      0.64      0.62      4000
        GFSK       0.68      0.64      0.66      4000
        PAM4       0.71      0.69      0.70      4000
       QAM16       0.55      0.55      0.55      4000
       QAM64       0.66      0.60      0.63      4000
        QPSK       0.54      0.58      0.56      4000
        WBFM       0.42      0.44      0.43      4000

    accuracy                           0.55     44000
   macro avg       0.55      0.55      0.55     44000
weighted avg       0.55      0.55      0.55     44000


# cnn1d

Overall accuracy: 0.5494
SNR -20.0 dB: Accuracy = 0.0951
SNR -18.0 dB: Accuracy = 0.0878
SNR -16.0 dB: Accuracy = 0.1087
SNR -14.0 dB: Accuracy = 0.1131
SNR -12.0 dB: Accuracy = 0.1341
SNR -10.0 dB: Accuracy = 0.2015
SNR -8.0 dB: Accuracy = 0.3191
SNR -6.0 dB: Accuracy = 0.4958
SNR -4.0 dB: Accuracy = 0.5926
SNR -2.0 dB: Accuracy = 0.6907
SNR 0.0 dB: Accuracy = 0.7593
SNR 2.0 dB: Accuracy = 0.7901
SNR 4.0 dB: Accuracy = 0.8245
SNR 6.0 dB: Accuracy = 0.8233
SNR 8.0 dB: Accuracy = 0.8170
SNR 10.0 dB: Accuracy = 0.8364
SNR 12.0 dB: Accuracy = 0.8382
SNR 14.0 dB: Accuracy = 0.8235
SNR 16.0 dB: Accuracy = 0.8035
SNR 18.0 dB: Accuracy = 0.8130

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.66      0.53      0.59      4000
      AM-DSB       0.51      0.64      0.57      4000
      AM-SSB       0.28      0.74      0.41      4000
        BPSK       0.71      0.60      0.65      4000
       CPFSK       0.80      0.60      0.69      4000
        GFSK       0.79      0.63      0.70      4000
        PAM4       0.87      0.66      0.75      4000
       QAM16       0.40      0.18      0.25      4000
       QAM64       0.53      0.59      0.56      4000
        QPSK       0.54      0.56      0.55      4000
        WBFM       0.60      0.30      0.40      4000

    accuracy                           0.55     44000
   macro avg       0.61      0.55      0.56     44000
weighted avg       0.61      0.55      0.56     44000

# cnn2d 

Overall accuracy: 0.4731
SNR -20.0 dB: Accuracy = 0.1001
SNR -18.0 dB: Accuracy = 0.0878
SNR -16.0 dB: Accuracy = 0.0859
SNR -14.0 dB: Accuracy = 0.1000
SNR -12.0 dB: Accuracy = 0.1176
SNR -10.0 dB: Accuracy = 0.1693
SNR -8.0 dB: Accuracy = 0.2962
SNR -6.0 dB: Accuracy = 0.4203
SNR -4.0 dB: Accuracy = 0.4648
SNR -2.0 dB: Accuracy = 0.5476
SNR 0.0 dB: Accuracy = 0.6104
SNR 2.0 dB: Accuracy = 0.6801
SNR 4.0 dB: Accuracy = 0.7155
SNR 6.0 dB: Accuracy = 0.7161
SNR 8.0 dB: Accuracy = 0.7186
SNR 10.0 dB: Accuracy = 0.7232
SNR 12.0 dB: Accuracy = 0.7612
SNR 14.0 dB: Accuracy = 0.7197
SNR 16.0 dB: Accuracy = 0.6894
SNR 18.0 dB: Accuracy = 0.7179

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.28      0.50      0.36      4000
      AM-DSB       0.53      0.60      0.57      4000
      AM-SSB       0.27      0.44      0.34      4000
        BPSK       0.55      0.57      0.56      4000
       CPFSK       0.56      0.57      0.57      4000
        GFSK       0.56      0.65      0.61      4000
        PAM4       0.82      0.60      0.69      4000
       QAM16       0.35      0.23      0.28      4000
       QAM64       0.55      0.51      0.53      4000
        QPSK       0.57      0.25      0.34      4000
        WBFM       0.56      0.27      0.36      4000

    accuracy                           0.47     44000
   macro avg       0.51      0.47      0.47     44000
weighted avg       0.51      0.47      0.47     44000


# complexnn

Overall accuracy: 0.5365
SNR -20.0 dB: Accuracy = 0.0866
SNR -18.0 dB: Accuracy = 0.0882
SNR -16.0 dB: Accuracy = 0.0971
SNR -14.0 dB: Accuracy = 0.1058
SNR -12.0 dB: Accuracy = 0.1270
SNR -10.0 dB: Accuracy = 0.1849
SNR -8.0 dB: Accuracy = 0.3196
SNR -6.0 dB: Accuracy = 0.4780
SNR -4.0 dB: Accuracy = 0.5756
SNR -2.0 dB: Accuracy = 0.6911
SNR 0.0 dB: Accuracy = 0.7470
SNR 2.0 dB: Accuracy = 0.7666
SNR 4.0 dB: Accuracy = 0.8007
SNR 6.0 dB: Accuracy = 0.8146
SNR 8.0 dB: Accuracy = 0.8009
SNR 10.0 dB: Accuracy = 0.8179
SNR 12.0 dB: Accuracy = 0.8099
SNR 14.0 dB: Accuracy = 0.8176
SNR 16.0 dB: Accuracy = 0.7955
SNR 18.0 dB: Accuracy = 0.7868

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.71      0.48      0.57      4000
      AM-DSB       0.53      0.48      0.51      4000
      AM-SSB       0.27      0.80      0.40      4000
        BPSK       0.71      0.61      0.65      4000
       CPFSK       0.87      0.59      0.70      4000
        GFSK       0.75      0.62      0.68      4000
        PAM4       0.88      0.65      0.75      4000
       QAM16       0.45      0.24      0.31      4000
       QAM64       0.54      0.53      0.53      4000
        QPSK       0.54      0.54      0.54      4000
        WBFM       0.45      0.37      0.41      4000

    accuracy                           0.54     44000
   macro avg       0.61      0.54      0.55     44000
weighted avg       0.61      0.54      0.55     44000


# transformer

Overall accuracy: 0.4786
SNR -20.0 dB: Accuracy = 0.0902
SNR -18.0 dB: Accuracy = 0.0840
SNR -16.0 dB: Accuracy = 0.0980
SNR -14.0 dB: Accuracy = 0.1018
SNR -12.0 dB: Accuracy = 0.1068
SNR -10.0 dB: Accuracy = 0.1398
SNR -8.0 dB: Accuracy = 0.2201
SNR -6.0 dB: Accuracy = 0.2969
SNR -4.0 dB: Accuracy = 0.3894
SNR -2.0 dB: Accuracy = 0.4732
SNR 0.0 dB: Accuracy = 0.5995
SNR 2.0 dB: Accuracy = 0.6809
SNR 4.0 dB: Accuracy = 0.7511
SNR 6.0 dB: Accuracy = 0.7743
SNR 8.0 dB: Accuracy = 0.7706
SNR 10.0 dB: Accuracy = 0.8034
SNR 12.0 dB: Accuracy = 0.8076
SNR 14.0 dB: Accuracy = 0.7916
SNR 16.0 dB: Accuracy = 0.7888
SNR 18.0 dB: Accuracy = 0.7828

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.58      0.41      0.48      4000
      AM-DSB       0.49      0.65      0.56      4000
      AM-SSB       0.22      0.94      0.36      4000
        BPSK       0.84      0.52      0.64      4000
       CPFSK       0.57      0.52      0.55      4000
        GFSK       0.70      0.41      0.52      4000
        PAM4       0.90      0.60      0.72      4000
       QAM16       0.53      0.12      0.20      4000
       QAM64       0.51      0.47      0.49      4000
        QPSK       0.89      0.41      0.56      4000
        WBFM       0.61      0.20      0.30      4000

    accuracy                           0.48     44000
   macro avg       0.62      0.48      0.49     44000
weighted avg       0.62      0.48      0.49     44000

# resnet + augment

## 1

Overall accuracy: 0.5934
SNR -20.0 dB: Accuracy = 0.0947
SNR -18.0 dB: Accuracy = 0.0882
SNR -16.0 dB: Accuracy = 0.1008
SNR -14.0 dB: Accuracy = 0.1117
SNR -12.0 dB: Accuracy = 0.1395
SNR -10.0 dB: Accuracy = 0.2042
SNR -8.0 dB: Accuracy = 0.3260
SNR -6.0 dB: Accuracy = 0.4945
SNR -4.0 dB: Accuracy = 0.6357
SNR -2.0 dB: Accuracy = 0.7688
SNR 0.0 dB: Accuracy = 0.8383
SNR 2.0 dB: Accuracy = 0.8736
SNR 4.0 dB: Accuracy = 0.8849
SNR 6.0 dB: Accuracy = 0.8965
SNR 8.0 dB: Accuracy = 0.8924
SNR 10.0 dB: Accuracy = 0.9026
SNR 12.0 dB: Accuracy = 0.9105
SNR 14.0 dB: Accuracy = 0.9061
SNR 16.0 dB: Accuracy = 0.8935
SNR 18.0 dB: Accuracy = 0.8827

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.63      0.56      0.60      4000
      AM-DSB       0.54      0.58      0.56      4000
      AM-SSB       0.29      0.60      0.39      4000
        BPSK       0.75      0.60      0.67      4000
       CPFSK       0.72      0.63      0.67      4000
        GFSK       0.70      0.65      0.68      4000
        PAM4       0.80      0.68      0.74      4000
       QAM16       0.66      0.61      0.63      4000
       QAM64       0.70      0.66      0.68      4000
        QPSK       0.62      0.58      0.60      4000
        WBFM       0.54      0.36      0.43      4000

    accuracy                           0.59     44000
   macro avg       0.63      0.59      0.60     44000
weighted avg       0.63      0.59      0.60     44000

## 2


Overall accuracy: 0.5997
SNR -20.0 dB: Accuracy = 0.0915
SNR -18.0 dB: Accuracy = 0.0798
SNR -16.0 dB: Accuracy = 0.1022
SNR -14.0 dB: Accuracy = 0.1266
SNR -12.0 dB: Accuracy = 0.1408
SNR -10.0 dB: Accuracy = 0.2162
SNR -8.0 dB: Accuracy = 0.3494
SNR -6.0 dB: Accuracy = 0.4971
SNR -4.0 dB: Accuracy = 0.6101
SNR -2.0 dB: Accuracy = 0.7533
SNR 0.0 dB: Accuracy = 0.8374
SNR 2.0 dB: Accuracy = 0.8921
SNR 4.0 dB: Accuracy = 0.9069
SNR 6.0 dB: Accuracy = 0.9121
SNR 8.0 dB: Accuracy = 0.9076
SNR 10.0 dB: Accuracy = 0.9017
SNR 12.0 dB: Accuracy = 0.9235
SNR 14.0 dB: Accuracy = 0.9151
SNR 16.0 dB: Accuracy = 0.9033
SNR 18.0 dB: Accuracy = 0.9027

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.72      0.54      0.62      4000
      AM-DSB       0.56      0.60      0.58      4000
      AM-SSB       0.28      0.78      0.41      4000
        BPSK       0.79      0.61      0.69      4000
       CPFSK       0.79      0.61      0.69      4000
        GFSK       0.72      0.64      0.68      4000
        PAM4       0.84      0.68      0.75      4000
       QAM16       0.66      0.59      0.62      4000
       QAM64       0.72      0.67      0.69      4000
        QPSK       0.78      0.54      0.64      4000
        WBFM       0.58      0.34      0.43      4000

    accuracy                           0.60     44000
   macro avg       0.68      0.60      0.62     44000
weighted avg       0.68      0.60      0.62     44000



# resnet + gpr

Overall accuracy: 0.6222
SNR -20.0 dB: Accuracy = 0.0947
SNR -18.0 dB: Accuracy = 0.1139
SNR -16.0 dB: Accuracy = 0.1260
SNR -14.0 dB: Accuracy = 0.1705
SNR -12.0 dB: Accuracy = 0.2293
SNR -10.0 dB: Accuracy = 0.3174
SNR -8.0 dB: Accuracy = 0.4278
SNR -6.0 dB: Accuracy = 0.5690
SNR -4.0 dB: Accuracy = 0.6810
SNR -2.0 dB: Accuracy = 0.7947
SNR 0.0 dB: Accuracy = 0.8553
SNR 2.0 dB: Accuracy = 0.8694
SNR 4.0 dB: Accuracy = 0.8830
SNR 6.0 dB: Accuracy = 0.9020
SNR 8.0 dB: Accuracy = 0.8933
SNR 10.0 dB: Accuracy = 0.9008
SNR 12.0 dB: Accuracy = 0.8989
SNR 14.0 dB: Accuracy = 0.8958
SNR 16.0 dB: Accuracy = 0.9033
SNR 18.0 dB: Accuracy = 0.8960

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.57      0.60      0.59      4000
      AM-DSB       0.51      0.50      0.50      4000
      AM-SSB       0.66      0.69      0.68      4000
        BPSK       0.67      0.65      0.66      4000
       CPFSK       0.63      0.66      0.64      4000
        GFSK       0.71      0.67      0.69      4000
        PAM4       0.76      0.72      0.74      4000
       QAM16       0.63      0.59      0.61      4000
       QAM64       0.70      0.73      0.71      4000
        QPSK       0.60      0.59      0.60      4000
        WBFM       0.43      0.46      0.44      4000

    accuracy                           0.62     44000
   macro avg       0.62      0.62      0.62     44000
weighted avg       0.62      0.62      0.62     44000




# resnet + gpr + augment

Overall accuracy: 0.6420
SNR -20.0 dB: Accuracy = 0.0915
SNR -18.0 dB: Accuracy = 0.1013
SNR -16.0 dB: Accuracy = 0.1143
SNR -14.0 dB: Accuracy = 0.1682
SNR -12.0 dB: Accuracy = 0.2311
SNR -10.0 dB: Accuracy = 0.3206
SNR -8.0 dB: Accuracy = 0.4654
SNR -6.0 dB: Accuracy = 0.5908
SNR -4.0 dB: Accuracy = 0.7111
SNR -2.0 dB: Accuracy = 0.8362
SNR 0.0 dB: Accuracy = 0.8827
SNR 2.0 dB: Accuracy = 0.9034
SNR 4.0 dB: Accuracy = 0.9130
SNR 6.0 dB: Accuracy = 0.9231
SNR 8.0 dB: Accuracy = 0.9177
SNR 10.0 dB: Accuracy = 0.9298
SNR 12.0 dB: Accuracy = 0.9388
SNR 14.0 dB: Accuracy = 0.9322
SNR 16.0 dB: Accuracy = 0.9272
SNR 18.0 dB: Accuracy = 0.9187

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.62      0.60      0.61      4000
      AM-DSB       0.52      0.55      0.54      4000
      AM-SSB       0.65      0.70      0.67      4000
        BPSK       0.69      0.65      0.67      4000
       CPFSK       0.68      0.65      0.67      4000
        GFSK       0.61      0.67      0.64      4000
        PAM4       0.71      0.73      0.72      4000
       QAM16       0.69      0.71      0.70      4000
       QAM64       0.81      0.76      0.78      4000
        QPSK       0.64      0.60      0.62      4000
        WBFM       0.45      0.42      0.44      4000

    accuracy                           0.64     44000
   macro avg       0.64      0.64      0.64     44000
weighted avg       0.64      0.64      0.64     44000

# complexnn + mod_relu

Overall accuracy: 0.5414
SNR -20.0 dB: Accuracy = 0.0942
SNR -18.0 dB: Accuracy = 0.0864
SNR -16.0 dB: Accuracy = 0.1017
SNR -14.0 dB: Accuracy = 0.1144
SNR -12.0 dB: Accuracy = 0.1489
SNR -10.0 dB: Accuracy = 0.2116
SNR -8.0 dB: Accuracy = 0.3320
SNR -6.0 dB: Accuracy = 0.4470
SNR -4.0 dB: Accuracy = 0.5635
SNR -2.0 dB: Accuracy = 0.6737
SNR 0.0 dB: Accuracy = 0.7300
SNR 2.0 dB: Accuracy = 0.7729
SNR 4.0 dB: Accuracy = 0.8128
SNR 6.0 dB: Accuracy = 0.8136
SNR 8.0 dB: Accuracy = 0.8005
SNR 10.0 dB: Accuracy = 0.8228
SNR 12.0 dB: Accuracy = 0.8405
SNR 14.0 dB: Accuracy = 0.8221
SNR 16.0 dB: Accuracy = 0.8035
SNR 18.0 dB: Accuracy = 0.8161
/home/test/.conda/envs/ljk2/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/test/.conda/envs/ljk2/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/test/.conda/envs/ljk2/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.48      0.57      0.52      4000
      AM-DSB       0.57      0.63      0.60      4000
      AM-SSB       0.27      0.86      0.41      4000
        BPSK       0.85      0.56      0.68      4000
       CPFSK       0.83      0.59      0.69      4000
        GFSK       0.77      0.65      0.70      4000
        PAM4       0.73      0.69      0.71      4000
       QAM16       0.00      0.00      0.00      4000
       QAM64       0.50      0.71      0.59      4000
        QPSK       0.79      0.47      0.59      4000
        WBFM       0.63      0.22      0.33      4000

    accuracy                           0.54     44000
   macro avg       0.58      0.54      0.53     44000
weighted avg       0.58      0.54      0.53     44000



# complexnn + relu

Overall accuracy: 0.5711
SNR -20.0 dB: Accuracy = 0.0960
SNR -18.0 dB: Accuracy = 0.0915
SNR -16.0 dB: Accuracy = 0.1097
SNR -14.0 dB: Accuracy = 0.1180
SNR -12.0 dB: Accuracy = 0.1417
SNR -10.0 dB: Accuracy = 0.1987
SNR -8.0 dB: Accuracy = 0.3269
SNR -6.0 dB: Accuracy = 0.4638
SNR -4.0 dB: Accuracy = 0.5976
SNR -2.0 dB: Accuracy = 0.7519
SNR 0.0 dB: Accuracy = 0.8085
SNR 2.0 dB: Accuracy = 0.8401
SNR 4.0 dB: Accuracy = 0.8479
SNR 6.0 dB: Accuracy = 0.8668
SNR 8.0 dB: Accuracy = 0.8579
SNR 10.0 dB: Accuracy = 0.8636
SNR 12.0 dB: Accuracy = 0.8646
SNR 14.0 dB: Accuracy = 0.8562
SNR 16.0 dB: Accuracy = 0.8483
SNR 18.0 dB: Accuracy = 0.8507

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.65      0.55      0.60      4000
      AM-DSB       0.46      0.74      0.57      4000
      AM-SSB       0.27      0.68      0.39      4000
        BPSK       0.72      0.61      0.66      4000
       CPFSK       0.81      0.61      0.70      4000
        GFSK       0.80      0.63      0.71      4000
        PAM4       0.88      0.67      0.76      4000
       QAM16       0.61      0.47      0.53      4000
       QAM64       0.62      0.51      0.56      4000
        QPSK       0.65      0.57      0.60      4000
        WBFM       0.53      0.25      0.34      4000

    accuracy                           0.57     44000
   macro avg       0.64      0.57      0.58     44000
weighted avg       0.64      0.57      0.58     44000


# complexnn + relu + gpr + augment

Overall accuracy: 0.6341
SNR -20.0 dB: Accuracy = 0.0956
SNR -18.0 dB: Accuracy = 0.1139
SNR -16.0 dB: Accuracy = 0.1330
SNR -14.0 dB: Accuracy = 0.1895
SNR -12.0 dB: Accuracy = 0.2383
SNR -10.0 dB: Accuracy = 0.3220
SNR -8.0 dB: Accuracy = 0.4883
SNR -6.0 dB: Accuracy = 0.6201
SNR -4.0 dB: Accuracy = 0.7290
SNR -2.0 dB: Accuracy = 0.8347
SNR 0.0 dB: Accuracy = 0.8586
SNR 2.0 dB: Accuracy = 0.8862
SNR 4.0 dB: Accuracy = 0.8905
SNR 6.0 dB: Accuracy = 0.9006
SNR 8.0 dB: Accuracy = 0.8897
SNR 10.0 dB: Accuracy = 0.9021
SNR 12.0 dB: Accuracy = 0.9068
SNR 14.0 dB: Accuracy = 0.8904
SNR 16.0 dB: Accuracy = 0.8855
SNR 18.0 dB: Accuracy = 0.8849

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.84      0.56      0.67      4000
      AM-DSB       0.50      0.71      0.59      4000
      AM-SSB       0.48      0.78      0.59      4000
        BPSK       0.85      0.63      0.73      4000
       CPFSK       0.90      0.62      0.74      4000
        GFSK       0.73      0.68      0.71      4000
        PAM4       0.86      0.71      0.78      4000
       QAM16       0.40      0.67      0.50      4000
       QAM64       0.58      0.79      0.67      4000
        QPSK       0.78      0.57      0.66      4000
        WBFM       0.73      0.24      0.36      4000

    accuracy                           0.63     44000
   macro avg       0.70      0.63      0.64     44000
weighted avg       0.70      0.63      0.64     44000


# complexnn + leaky relu

Overall accuracy: 0.5618
SNR -20.0 dB: Accuracy = 0.0888
SNR -18.0 dB: Accuracy = 0.0854
SNR -16.0 dB: Accuracy = 0.0947
SNR -14.0 dB: Accuracy = 0.1140
SNR -12.0 dB: Accuracy = 0.1489
SNR -10.0 dB: Accuracy = 0.2180
SNR -8.0 dB: Accuracy = 0.3333
SNR -6.0 dB: Accuracy = 0.5135
SNR -4.0 dB: Accuracy = 0.6196
SNR -2.0 dB: Accuracy = 0.7491
SNR 0.0 dB: Accuracy = 0.7787
SNR 2.0 dB: Accuracy = 0.8170
SNR 4.0 dB: Accuracy = 0.8222
SNR 6.0 dB: Accuracy = 0.8411
SNR 8.0 dB: Accuracy = 0.8216
SNR 10.0 dB: Accuracy = 0.8378
SNR 12.0 dB: Accuracy = 0.8363
SNR 14.0 dB: Accuracy = 0.8401
SNR 16.0 dB: Accuracy = 0.8150
SNR 18.0 dB: Accuracy = 0.8387

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.61      0.57      0.59      4000
      AM-DSB       0.57      0.60      0.58      4000
      AM-SSB       0.28      0.77      0.41      4000
        BPSK       0.75      0.61      0.67      4000
       CPFSK       0.75      0.62      0.68      4000
        GFSK       0.70      0.68      0.69      4000
        PAM4       0.86      0.68      0.76      4000
       QAM16       0.45      0.36      0.40      4000
       QAM64       0.54      0.42      0.47      4000
        QPSK       0.74      0.54      0.63      4000
        WBFM       0.57      0.33      0.42      4000

    accuracy                           0.56     44000
   macro avg       0.62      0.56      0.57     44000
weighted avg       0.62      0.56      0.57     44000



# complexnn + leaky relu + gpr

Overall accuracy: 0.6140
SNR -20.0 dB: Accuracy = 0.0902
SNR -18.0 dB: Accuracy = 0.1148
SNR -16.0 dB: Accuracy = 0.1470
SNR -14.0 dB: Accuracy = 0.1796
SNR -12.0 dB: Accuracy = 0.2539
SNR -10.0 dB: Accuracy = 0.3602
SNR -8.0 dB: Accuracy = 0.4732
SNR -6.0 dB: Accuracy = 0.6032
SNR -4.0 dB: Accuracy = 0.7017
SNR -2.0 dB: Accuracy = 0.7994
SNR 0.0 dB: Accuracy = 0.8293
SNR 2.0 dB: Accuracy = 0.8455
SNR 4.0 dB: Accuracy = 0.8465
SNR 6.0 dB: Accuracy = 0.8636
SNR 8.0 dB: Accuracy = 0.8520
SNR 10.0 dB: Accuracy = 0.8550
SNR 12.0 dB: Accuracy = 0.8748
SNR 14.0 dB: Accuracy = 0.8666
SNR 16.0 dB: Accuracy = 0.8478
SNR 18.0 dB: Accuracy = 0.8543

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.64      0.59      0.61      4000
      AM-DSB       0.51      0.66      0.57      4000
      AM-SSB       0.67      0.71      0.69      4000
        BPSK       0.70      0.66      0.68      4000
       CPFSK       0.63      0.65      0.64      4000
        GFSK       0.66      0.69      0.68      4000
        PAM4       0.75      0.73      0.74      4000
       QAM16       0.50      0.51      0.50      4000
       QAM64       0.59      0.62      0.60      4000
        QPSK       0.61      0.60      0.61      4000
        WBFM       0.51      0.33      0.40      4000

    accuracy                           0.61     44000
   macro avg       0.61      0.61      0.61     44000
weighted avg       0.61      0.61      0.61     44000

# resnet + gpr + augment + v2

Overall accuracy: 0.6437
SNR -20.0 dB: Accuracy = 0.0893
SNR -18.0 dB: Accuracy = 0.0906
SNR -16.0 dB: Accuracy = 0.1255
SNR -14.0 dB: Accuracy = 0.1850
SNR -12.0 dB: Accuracy = 0.2164
SNR -10.0 dB: Accuracy = 0.3376
SNR -8.0 dB: Accuracy = 0.4530
SNR -6.0 dB: Accuracy = 0.5819
SNR -4.0 dB: Accuracy = 0.7218
SNR -2.0 dB: Accuracy = 0.8267
SNR 0.0 dB: Accuracy = 0.8794
SNR 2.0 dB: Accuracy = 0.9051
SNR 4.0 dB: Accuracy = 0.9195
SNR 6.0 dB: Accuracy = 0.9341
SNR 8.0 dB: Accuracy = 0.9287
SNR 10.0 dB: Accuracy = 0.9348
SNR 12.0 dB: Accuracy = 0.9351
SNR 14.0 dB: Accuracy = 0.9313
SNR 16.0 dB: Accuracy = 0.9335
SNR 18.0 dB: Accuracy = 0.9200

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.62      0.61      0.61      4000
      AM-DSB       0.52      0.56      0.54      4000
      AM-SSB       0.65      0.70      0.68      4000
        BPSK       0.72      0.65      0.68      4000
       CPFSK       0.70      0.65      0.68      4000
        GFSK       0.62      0.67      0.64      4000
        PAM4       0.71      0.73      0.72      4000
       QAM16       0.67      0.73      0.70      4000
       QAM64       0.81      0.75      0.78      4000
        QPSK       0.60      0.60      0.60      4000
        WBFM       0.48      0.43      0.45      4000

    accuracy                           0.64     44000
   macro avg       0.64      0.64      0.64     44000
weighted avg       0.64      0.64      0.64     44000

# resnet + gpr + augment + v3

Overall accuracy: 0.6420
SNR -20.0 dB: Accuracy = 0.0906
SNR -18.0 dB: Accuracy = 0.1041
SNR -16.0 dB: Accuracy = 0.1246
SNR -14.0 dB: Accuracy = 0.1578
SNR -12.0 dB: Accuracy = 0.2316
SNR -10.0 dB: Accuracy = 0.3160
SNR -8.0 dB: Accuracy = 0.4562
SNR -6.0 dB: Accuracy = 0.5957
SNR -4.0 dB: Accuracy = 0.7129
SNR -2.0 dB: Accuracy = 0.8202
SNR 0.0 dB: Accuracy = 0.8761
SNR 2.0 dB: Accuracy = 0.9085
SNR 4.0 dB: Accuracy = 0.9106
SNR 6.0 dB: Accuracy = 0.9336
SNR 8.0 dB: Accuracy = 0.9287
SNR 10.0 dB: Accuracy = 0.9289
SNR 12.0 dB: Accuracy = 0.9286
SNR 14.0 dB: Accuracy = 0.9335
SNR 16.0 dB: Accuracy = 0.9290
SNR 18.0 dB: Accuracy = 0.9271

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.59      0.61      0.60      4000
      AM-DSB       0.51      0.55      0.53      4000
      AM-SSB       0.63      0.70      0.66      4000
        BPSK       0.67      0.65      0.66      4000
       CPFSK       0.67      0.65      0.66      4000
        GFSK       0.70      0.67      0.68      4000
        PAM4       0.72      0.73      0.72      4000
       QAM16       0.69      0.70      0.69      4000
       QAM64       0.78      0.78      0.78      4000
        QPSK       0.61      0.60      0.61      4000
        WBFM       0.48      0.41      0.44      4000

    accuracy                           0.64     44000
   macro avg       0.64      0.64      0.64     44000
weighted avg       0.64      0.64      0.64     44000


# resnet + gpr + augment + v4

Overall accuracy: 0.6420
SNR -20.0 dB: Accuracy = 0.1023
SNR -18.0 dB: Accuracy = 0.0910
SNR -16.0 dB: Accuracy = 0.1125
SNR -14.0 dB: Accuracy = 0.1660
SNR -12.0 dB: Accuracy = 0.2181
SNR -10.0 dB: Accuracy = 0.3229
SNR -8.0 dB: Accuracy = 0.4640
SNR -6.0 dB: Accuracy = 0.5850
SNR -4.0 dB: Accuracy = 0.7156
SNR -2.0 dB: Accuracy = 0.8211
SNR 0.0 dB: Accuracy = 0.8809
SNR 2.0 dB: Accuracy = 0.9072
SNR 4.0 dB: Accuracy = 0.9186
SNR 6.0 dB: Accuracy = 0.9295
SNR 8.0 dB: Accuracy = 0.9255
SNR 10.0 dB: Accuracy = 0.9384
SNR 12.0 dB: Accuracy = 0.9397
SNR 14.0 dB: Accuracy = 0.9331
SNR 16.0 dB: Accuracy = 0.9241
SNR 18.0 dB: Accuracy = 0.9209

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.60      0.60      0.60      4000
      AM-DSB       0.49      0.58      0.53      4000
      AM-SSB       0.66      0.71      0.68      4000
        BPSK       0.67      0.65      0.66      4000
       CPFSK       0.67      0.65      0.66      4000
        GFSK       0.66      0.66      0.66      4000
        PAM4       0.73      0.73      0.73      4000
       QAM16       0.70      0.72      0.71      4000
       QAM64       0.81      0.76      0.78      4000
        QPSK       0.61      0.61      0.61      4000
        WBFM       0.46      0.40      0.43      4000

    accuracy                           0.64     44000
   macro avg       0.64      0.64      0.64     44000
weighted avg       0.64      0.64      0.64     44000


# lightweight hybrid complex !!

patience_lr=2, patience_es=30
factor=0.5

Overall accuracy: 0.6537
SNR -20.0 dB: Accuracy = 0.0965
SNR -18.0 dB: Accuracy = 0.1008
SNR -16.0 dB: Accuracy = 0.1321
SNR -14.0 dB: Accuracy = 0.1931
SNR -12.0 dB: Accuracy = 0.2472
SNR -10.0 dB: Accuracy = 0.3505
SNR -8.0 dB: Accuracy = 0.5062
SNR -6.0 dB: Accuracy = 0.6405
SNR -4.0 dB: Accuracy = 0.7784
SNR -2.0 dB: Accuracy = 0.8573
SNR 0.0 dB: Accuracy = 0.8780
SNR 2.0 dB: Accuracy = 0.9144
SNR 4.0 dB: Accuracy = 0.9139
SNR 6.0 dB: Accuracy = 0.9263
SNR 8.0 dB: Accuracy = 0.9154
SNR 10.0 dB: Accuracy = 0.9153
SNR 12.0 dB: Accuracy = 0.9337
SNR 14.0 dB: Accuracy = 0.9227
SNR 16.0 dB: Accuracy = 0.9135
SNR 18.0 dB: Accuracy = 0.9125

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.83      0.58      0.68      4000
      AM-DSB       0.50      0.71      0.59      4000
      AM-SSB       0.38      0.84      0.53      4000
        BPSK       0.77      0.65      0.70      4000
       CPFSK       0.87      0.64      0.74      4000
        GFSK       0.81      0.67      0.73      4000
        PAM4       0.81      0.72      0.76      4000
       QAM16       0.60      0.71      0.65      4000
       QAM64       0.78      0.79      0.79      4000
        QPSK       0.82      0.58      0.68      4000
        WBFM       0.58      0.29      0.39      4000

    accuracy                           0.65     44000
   macro avg       0.71      0.65      0.66     44000
weighted avg       0.71      0.65      0.66     44000


# lightweight transition !!
Overall accuracy: 0.6293
SNR -20.0 dB: Accuracy = 0.0951
SNR -18.0 dB: Accuracy = 0.1022
SNR -16.0 dB: Accuracy = 0.1232
SNR -14.0 dB: Accuracy = 0.1728
SNR -12.0 dB: Accuracy = 0.2231
SNR -10.0 dB: Accuracy = 0.3017
SNR -8.0 dB: Accuracy = 0.4319
SNR -6.0 dB: Accuracy = 0.5650
SNR -4.0 dB: Accuracy = 0.7111
SNR -2.0 dB: Accuracy = 0.8192
SNR 0.0 dB: Accuracy = 0.8634
SNR 2.0 dB: Accuracy = 0.8963
SNR 4.0 dB: Accuracy = 0.9045
SNR 6.0 dB: Accuracy = 0.9135
SNR 8.0 dB: Accuracy = 0.9002
SNR 10.0 dB: Accuracy = 0.9121
SNR 12.0 dB: Accuracy = 0.9175
SNR 14.0 dB: Accuracy = 0.9070
SNR 16.0 dB: Accuracy = 0.9020
SNR 18.0 dB: Accuracy = 0.9014

Classification Report:
              precision    recall  f1-score   support

        8PSK       0.73      0.56      0.63      4000
      AM-DSB       0.46      0.67      0.55      4000
      AM-SSB       0.59      0.73      0.65      4000
        BPSK       0.83      0.62      0.71      4000
       CPFSK       0.78      0.62      0.69      4000
        GFSK       0.76      0.64      0.69      4000
        PAM4       0.56      0.75      0.64      4000
       QAM16       0.56      0.62      0.59      4000
       QAM64       0.67      0.81      0.74      4000
        QPSK       0.63      0.60      0.61      4000
        WBFM       0.54      0.31      0.39      4000

    accuracy                           0.63     44000
   macro avg       0.65      0.63      0.63     44000
weighted avg       0.65      0.63      0.63     44000

# complexnn + relu !!

Epoch 1: val_accuracy improved from -inf to 0.57459, saving model to ../output/models/complex_nn_model.keras
s                                                                                                          lo
4813/4813 ━━━━━━━━━━━━━━━━━━━━ 34s 5ms/step - accuracy: 0.4992 - loss: 1.2476 - val_accuracy: 0.5746 - val_
loss: 1.0416 - learning_rate: 0.0010                                                
Epoch 2/500                                                                         
4795/4813 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5857 - loss: 1.0132                               er
Epoch 2: val_accuracy improved from 0.57459 to 0.58945, saving model to ../output/models/complex_nn_model.k
eras                                                                                                       lo
4813/4813 ━━━━━━━━━━━━━━━━━━━━ 16s 3ms/step - accuracy: 0.5858 - loss: 1.0132 - val_accuracy: 0.5895 - val_
loss: 1.0147 - learning_rate: 0.0010                                                
Epoch 3/500                                                                         
4803/4813 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.5981 - loss: 0.9838                               er
Epoch 3: val_accuracy improved from 0.58945 to 0.60591, saving model to ../output/models/complex_nn_model.k
eras                                                                                                       lo
4813/4813 ━━━━━━━━━━━━━━━━━━━━ 16s 3ms/step - accuracy: 0.5981 - loss: 0.9838 - val_accuracy: 0.6059 - val_
loss: 0.9763 - learning_rate: 0.0010                                                
Epoch 4/500                                                                         
4812/4813 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6071 - loss: 0.9606                               er
Epoch 4: val_accuracy improved from 0.60591 to 0.60805, saving model to ../output/models/complex_nn_model.k
eras                                                                                                       lo
4813/4813 ━━━━━━━━━━━━━━━━━━━━ 17s 3ms/step - accuracy: 0.6071 - loss: 0.9606 - val_accuracy: 0.6080 - val_
loss: 0.9642 - learning_rate: 0.0010                                                
Epoch 5/500                                                                         
4786/4813 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.6146 - loss: 0.9467                               er
Epoch 5: val_accuracy improved from 0.60805 to 0.61473, saving model to ../output/models/complex_nn_model.k
eras                                                                                                       lo
4813/4813 ━━━━━━━━━━━━━━━━━━━━ 12s 3ms/step - accuracy: 0.6146 - loss: 0.9467 - val_accuracy: 0.6147 - val_
loss: 0.9528 - learning_rate: 0.0010   

