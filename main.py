import os
import cv2
import json
import shutil
import numpy as np
import argparse

from retinaface import RetinaFace
from face_recognition import ArcFace
from sklearn.cluster import DBSCAN

def get_faces_dict(input_dir, model):
    '''
    Takes input directory path and face recognition path and returns dictionary with face feature vector
    '''
    faces_dict = {}
    for file in os.listdir(input_dir):
    #     print(os.path.join(input_dir,file))
        file_name = file.split('.')[0]
        file_ext = '.' + file.split('.')[-1]
        img_path = os.path.join(input_dir,file)
        faces = RetinaFace.detect_faces(img_path)
        if type(faces) is dict:
            img = cv2.imread(img_path)
            for key in faces.keys():
                identity = faces[key]
                facial_area = identity["facial_area"]
                y1 = facial_area[1]
                y2 = facial_area[3]
                x1 = facial_area[0]
                x2 = facial_area[2]
                face = img[y1:y2, x1:x2]
                face = cv2.resize(face, (112,112))
                face = np.expand_dims(face, axis=0)
        #         embs.append(model.predict(face)[0])
                faces_dict[file_name + '_' + key + file_ext] = model.predict(face)[0]
    return faces_dict

def get_images(list):
    images_list = []
    for face in list:
        image_name = face.split('_')[0] + '.' + face.split('.')[-1]
        images_list.append(image_name)
    return images_list

def main(args):
    # Load face recognition model
    model = ArcFace.loadModel()
    
    faces_dict = get_faces_dict(args.input_dir, model)

    embs = []
    faces_list = []
    for key, value in faces_dict.items():
        faces_list.append(key)
        embs.append(value)
    embs_arr = np.array(embs)

    db_default = DBSCAN(eps = args.eps, min_samples = args.min_samples, metric='cosine').fit(embs_arr)
    labels = db_default.labels_
    print('Labels =  ', labels)
    
    faces_arr = np.array(faces_list)
    images_dict = {}
    for grp in set(labels):
        ii = np.where(np.array(labels) == grp)[0]
        images_dict[int(grp)] = get_images(faces_arr[ii].tolist())

    with open('clusters.txt', 'w') as f:
        f.write(json.dumps(images_dict, indent=2))

    if args.make_dir:
        for key, values in images_dict.items():
            for value in values:
                dst_path = os.path.join('output', str(key))
                if not os.path.isdir(dst_path):
                    os.makedirs(dst_path)
                source_path = os.path.join(args.input_dir, value)
                shutil.copy(source_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="images", help="path to input directory")
    parser.add_argument("--eps", type=float, default=0.07)
    parser.add_argument("--min_samples", type=int, default=1)

    parser.add_argument('--make-dir', action='store_true')
    # parser.add_argument('--no-make-dir', dest='make-dir', action='store_false')
    parser.set_defaults(make_dir=False)
    args = parser.parse_args()
    main(args)