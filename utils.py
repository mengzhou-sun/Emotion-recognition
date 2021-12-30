import numpy as np
from numpy.random import default_rng
import math



def PrRec(out, lbl):
    size = out.shape[0]
    confidence = np.expand_dims(out.T, axis=1)
    lbl = np.expand_dims(lbl.T, axis=1)
    confidence += np.random.rand(size, 1) * (10 ** (-10))

    confidence_ = -1 * confidence
    ind = np.argsort(confidence_, axis=0)
    confidence_ = np.take_along_axis(confidence_, ind, axis=0)
    C = np.take_along_axis(lbl, ind, axis=0)
    n = len(C)

    REL = np.sum(C)
    if n > 0:
        RETREL = np.cumsum(C)
        RET = np.arange(1, n + 1).T
    else:
        RETREL = 0
        RET = 1

    precision = (100 * RETREL) / RET
    recall = (100 * RETREL) / REL
    th = -1 * confidence_

    mrec = np.append(recall, 100)
    mrec = np.insert(mrec, 0, 0)
    mpre = np.append(precision, 0)
    mpre = np.insert(mpre, 0, 0)

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:-1] != mrec[0:-2])[0]
    i += 1
    temp = 0
    for j in range(len(i)):
        ind = i[j]
        temp += (mrec[ind] - mrec[ind - 1]) * mpre[ind]

    avg_precision = temp / 100

    return [recall], [precision], [th], avg_precision


def modified_AP(output, labels):
    '''
    output --> (#categories, #samples)
    labels --> (#categories, #samples)
    '''
    categories = labels.shape[0]
    recall = []
    precision = []
    th = []
    avg_precision = []
    for i in range(categories):
        r, p, t, a = PrRec(output[i, :], labels[i, :])
        recall.append(r)
        precision.append(p)
        th.append(t)
        if a < 0:
            a = 0
        avg_precision.append(a)

    return avg_precision


def sort_by_image_index(image_index, array, key):
    if key == "coco_url" or key == "saliency_path" or key == "file_name":
        output = np.zeros((np.shape(array)[0], np.array(array[0][1]).flatten().shape[0]), dtype=object)
    else:
        output = np.zeros((np.shape(array)[0], np.array(array[0][1]).flatten().shape[0]))
    for i in range(0, len(image_index)):
        for j in array:
            if j[0] == image_index[i]:
                try:
                    output[i, :] = np.array(j[1]).flatten()
                except:
                    output[i, :] = j[1]

    return output


def organize_array(array, key, origin="annotations", mode="train"):
    array = np.array(array[origin])
    output = np.zeros((len(array), 2), dtype=object)
    for i in range(len(array)):
        if key == "coco_url" or key == "saliency_path" or key == "file_name":
            output[i, 0] = array[i]["id"]
        elif key != "emotion":
            output[i, 0] = array[i]["image_id"]
        if key == "emotions":
            output[i, 1] = one_hot_emotions(array[i][key])
        elif key == "saliency_path":
            output[i, 1] = mode + "_saliency/" + array[i]["file_name"]
        else:
            output[i, 1] = array[i][key]

    sorted_index_file = np.sort(output[:, 0])
    output = sort_by_image_index(sorted_index_file, output, key)
    return output


def one_hot_emotions(emotion_list):
    emotion_list = emotion_list[0]
    out = np.zeros(26)
    for emotion in emotion_list:
        emotion = emotion.strip()
        if emotion.lower() == "Affection".lower():
            emotion_id = 0
        elif emotion.lower() == "Anger".lower():
            emotion_id = 1
        elif emotion.lower() == "Annoyance".lower():
            emotion_id = 2
        elif emotion.lower() == "Anticipation".lower():
            emotion_id = 3
        elif emotion.lower() == "Aversion".lower():
            emotion_id = 4
        elif emotion.lower() == "Confidence".lower():
            emotion_id = 5
        elif emotion.lower() == "Disapproval".lower():
            emotion_id = 6
        elif emotion.lower() == "Disconnection".lower():
            emotion_id = 7
        elif emotion.lower() == "Disquietment".lower():
            emotion_id = 8
        elif emotion.lower() == "Doubt/Confusion".lower() or emotion.lower() == "Doubt".lower() or emotion.lower() == "Confusion".lower():
            emotion_id = 9
        elif emotion.lower() == "Embarrassment".lower():
            emotion_id = 10
        elif emotion.lower() == "Engagement".lower():
            emotion_id = 11
        elif emotion.lower() == "Esteem".lower():
            emotion_id = 12
        elif emotion.lower() == "Excitement".lower():
            emotion_id = 13
        elif emotion.lower() == "Fatigue".lower():
            emotion_id = 14
        elif emotion.lower() == "Fear".lower():
            emotion_id = 15
        elif emotion.lower() == "Happiness".lower():
            emotion_id = 16
        elif emotion.lower() == "Pain".lower():
            emotion_id = 17
        elif emotion.lower() == "Peace".lower():
            emotion_id = 18
        elif emotion.lower() == "Pleasure".lower():
            emotion_id = 19
        elif emotion.lower() == "Sadness".lower():
            emotion_id = 20
        elif emotion.lower() == "Sensitivity".lower():
            emotion_id = 21
        elif emotion.lower() == "Suffering".lower():
            emotion_id = 22
        elif emotion.lower() == "Surprise".lower():
            emotion_id = 23
        elif emotion.lower() == "Sympathy".lower():
            emotion_id = 24
        elif emotion.lower() == "Yearning".lower():
            emotion_id = 25

        out[emotion_id] = 1

    return out


def extract_faces(image, eye_x, eye_y, attention_full):
    img = image.copy()

    e = [eye_x, eye_y]
    # coordinates of the eyes

    alpha = 0.3
    w_x = int(math.floor(alpha * img.shape[1]))
    w_y = int(math.floor(alpha * img.shape[0]))

    if w_x % 2 == 0:
        w_x = w_x + 1

    if w_y % 2 == 0:
        w_y = w_y + 1

    im_face = np.ones((w_y, w_x, 3))
    im_face[:, :, 0] = 123 * np.ones((w_y, w_x))
    im_face[:, :, 1] = 117 * np.ones((w_y, w_x))
    im_face[:, :, 2] = 104 * np.ones((w_y, w_x))

    center = [math.floor(e[0] * img.shape[1]), math.floor(e[1] * img.shape[0])]
    d_x = math.floor((w_x - 1) / 2)
    d_y = math.floor((w_y - 1) / 2)

    bottom_x = center[0] - d_x - 1
    delta_b_x = 0
    if bottom_x < 0:
        delta_b_x = 1 - bottom_x
        bottom_x = 0

    top_x = center[0] + d_x - 1
    delta_t_x = w_x - 1
    if top_x > img.shape[1] - 1:
        delta_t_x = w_x - (top_x - img.shape[1] + 1)
        top_x = img.shape[1] - 1

    bottom_y = center[1] - d_y - 1
    delta_b_y = 0
    if bottom_y < 0:
        delta_b_y = 1 - bottom_y
        bottom_y = 0

    top_y = center[1] + d_y - 1
    delta_t_y = w_y - 1
    if top_y > img.shape[0] - 1:
        delta_t_y = w_y - (top_y - img.shape[0] + 1)
        top_y = img.shape[0] - 1

    topx = top_x
    if len(img.shape) == 3:
        x = img[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1), :]
    else:
        x = img[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1)]

    attention = attention_full.copy()
    # attention[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1)] = 0

    return x, attention

