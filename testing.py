import cv2
import numpy as np
import mediapipe as mp
import modules.config as config
import modules.functions as functions
from keras.models import load_model
import translators as ts
from PIL import Image,ImageDraw,ImageFont



def executable(lang,op_lang):

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    actions = functions.generate_actions(lang)
    model = load_model('C:/Users/rikhi/projectF/FinalProject/'+lang+'model.h5')
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    print("in executable",op_lang)
    if op_lang=='hi':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/TiroDevanagariHindi-Regular.ttf',30)
    elif op_lang=='te':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/NotoSansTelugu-VariableFont_wdth,wght.ttf',30)
    elif op_lang=='ta':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/NotoSansTamil-VariableFont_wdth,wght.ttf',30)
    elif op_lang=='en':
        font = ImageFont.truetype('C:/Users/rikhi/projectF/FinalProject/fonts/IMFellEnglish-Regular.ttf',30)
    with config.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            flag=0

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = functions.mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            functions.draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = functions.extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:15]
            
            if len(sequence) == 15:

                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                '''print(actions[np.argmax(res)])'''
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                
                # Viz probabilities
                # image = functions.prob_viz(res, actions, image)
                

                x = ' '.join(sentence)
                if op_lang!='en':
                    translated = ts.translate_text(x,from_language = 'en',to_language = op_lang)
                else:
                    translated = x
                # print("x= ",x)
                # print("translated= {} type={}".format(translated,type(translated)))
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((50, 80),  translated, font = font)
                op_lang_img = np.array(img_pil)
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.imshow('OpenCV Feed',op_lang_img)
                flag=1
                
                # cv2.putText(image, ' '.join(sentence), (3,30), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.imwrite(image,op_lang_img)
                
                
            # Show to screen
            # cv2.imshow('output',op_lang_img)
            if flag==0:
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.imshow('OpenCV Feed', image)
            

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# executable('bsl')