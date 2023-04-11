# FinalProject

Load model

https://www.projectpro.io/recipes/save-trained-model-in-python

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
=\\\\\\\\ASL: A-Z,hello,thanks,sorry,please,me,bye,yes,no,you,help [DONE]
BSL:hello,thanks,sorry,me,bye,yes,no,you,help
ISL:hello,thanks,sorry,please,namaste,me,bye,yes,no,you,help


ACCURACY RESULT TABULATION

Metric          ASL         BSL         ISL     FSL         model details

categorical                 
accuracy        1.0           1.0         1.0     1.0        no dropouts, input-64,lstm-128,64,64,dense-64,32, relu

                0.113                                            input=64, lstm=64,dropout=0.2,dense-64,32,relu

                0.127                                            input=64, lstm=64,dropout=0.2,dense-64,32,sigmoid

                1.0                                               input=64, lstm=64, dropout=0.2,dense-64,32,sigmoid

            1.0                                                input=64,lstm=128,64,tanh no dropout,dense-64,32,relu

        0.95                                            input=64,lstm=128,64,tanh,dropout=0.2,dense-64,32,sigmoid

        0.11                                      input=64,lstm=128,64,tanh,dropout=0.2,dense-64,32,sigmoid regularization in dense layers-L1,L2


        0.92 (ASL)                                      input-64, lstm-64 tanh, dropout=0.2, dense-32 sigmoid
<!-- buckets - 4 (lang)
in each bucket:

all signs as folders - 
in each sign - all videos as folders
in each video folder - all npy files -->

if op_lang=='hi':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/TiroDevanagariHindi-Regular.ttf',30)
    elif op_lang=='te':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/NotoSansTelugu-VariableFont_wdth,wght.ttf',30)
    elif op_lang=='ta':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/NotoSansTamil-VariableFont_wdth,wght.ttf',30)
    elif op_lang=='en':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/IMFellEnglish-Regular.ttf',30)