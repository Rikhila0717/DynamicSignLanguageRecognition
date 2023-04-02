# FinalProject

Load model

https://www.projectpro.io/recipes/save-trained-model-in-python

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

ASL: A-Z,hello,thanks,sorry,please,me,bye,yes,no,you,help [DONE]
BSL:hello,thanks,sorry,me,bye,yes,no,you,help
ISL:hello,thanks,sorry,please,me,bye,yes,no,you,help


ACCURACY RESULT TABULATION

Metric          ASL         BSL         ISL     FSL         model details

categorical                 
accuracy        1.0           1.0         1.0     1.0        no dropouts, input-64,lstm-128,64,64,dense-64,32, relu

                0.113                                            input=64, lstm=64,dropout=0.2,dense-64,32,relu

                0.127                                            input=64, lstm=64,dropout=0.2,dense-64,32,sigmoid

                1.0                                               input=64, lstm=64, dropout=0.2,dense-64,32,sigmoid