import pickle
import numpy as np
import os
import io
import keras
import cv2

import face_recognition

def load_image(path):
    
    image = face_recognition.load_image_file('{}/image.png'.format(path))

    with open('{}/landmarks.pickle'.format(path), 'rb') as landmarks_p:
        landmarks = pickle.load(landmarks_p)
        
    x = [*get_inputs(image, landmarks, crash=True)]

    with open('{}/target.pickle'.format(path), 'rb') as target_p:
        y = pickle.load(target_p)
    
    return x, y

def load_data_subset(path):
    folders = set(os.listdir(path)) - {'.ipynb_checkpoints'}
    x_data, y_data = [[], [], [], []], []
    for folder in folders:
        x, y = load_image('{}/{}'.format(path, folder))
        for i in range(4):
            x_data[i].append(x[i])
        y_data.append(y)
    
    return [*map(np.array, x_data)], np.array(y_data)
        

def load_data():
    return load_data_subset('data/train'), load_data_subset('data/test')
    


def center(landmarks, feature):
    """
    Find the pixel center of a given feature from face_recognition landmarks
    
    :param landmarks: landmarks generated with face_recognition.face_landmarks
    :param feature: the relevant facial feature. Called 'feature' and not 'landmark' to avoid confusion
    :return: pixel center
    """
    return np.mean(landmarks[0][feature], axis=0)
    

def crop_part(image, landmark, landmarks=None, scale=(64, 64)):
    if landmarks is None:
        landmarks = face_recognition.face_landmarks(image)
        
    scale = tuple(map(int, scale))
    
    # calculate top-left corner of crop area
    pos = center(landmarks, landmark) - np.array([scale[0], scale[0]]) / 2
    x, y = tuple(map(int, pos))
    
    # crop image using slicing
    out = image[y:y+scale[0]].swapaxes(0, 1)[x:x+scale[0]].swapaxes(0, 1)
    
    # if necessary, rescale image before outputting
    if scale[0] != scale[1]:
        out = cv2.resize(out, dsize=(scale[1], scale[1]), interpolation=cv2.INTER_CUBIC)
    
    return out

def get_inputs(image, landmarks=None, crash=False, out_size=128):
    """
    Given an image, returns cropped areas and data relevant to eye tracking.
    
    :param image: image to search
    :param crash: if the function should crash or return None given an invalid image
    :return: left_eye, right_eye, face, aux_data
    """
    
    if landmarks is None:
        landmarks = face_recognition.face_landmarks(image)
    
    if landmarks or crash:
        # calculate distance between center of eyes to establish relative scale
        dist = np.linalg.norm(center(landmarks, 'left_eye') - center(landmarks, 'right_eye'))

        try:
            left_eye  = crop_part(image, 'left_eye', landmarks, scale=(dist*0.6, out_size))
            right_eye = crop_part(image, 'right_eye', landmarks, scale=(dist*0.6, out_size))
            face      = crop_part(image, 'nose_tip', landmarks, scale=(dist*2, out_size))
        except:
            if crash:
                assert False
            return None

        # flip right to maximize similarity
        right_eye = np.fliplr(right_eye)

        # calculate some relevant data: 
        #   distance between eyes -- scale of face
        #   all points in the landmark data -- abstract face shape view
        aux_data = [dist, *np.array(sum(landmarks[0].values(), [])).flat]

        return left_eye, right_eye, face, aux_data
    else:
        return None

# I'm like 60% sure this increases framerate
#def jpeg_array_to_img(array, fmt='jpeg'):
    #f = io.BytesIO()
    #keras.preprocessing.image.array_to_img(array).save(f, fmt)
    #return IPython.display.Image(data=f.getvalue())
