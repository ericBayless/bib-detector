import sys
import h5py
import cv2 as cv

def get_img_name(f, idx=0):
    """
    Adapted from: from https://www.vitaarca.net/post/tech/access_svhn_data_in_python/
    """
    names = f['digitStruct/name']
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)

def get_img_boxes(f, idx=0):
    """
    Adapted from:  from https://www.vitaarca.net/post/tech/access_svhn_data_in_python/
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    bboxs = f['digitStruct/bbox']
    box = f[bboxs[idx][0]]
    meta = { key : [] for key in box.keys()}

    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta

def create_annot_file(f, path, idx=0):
    # get image name and bounding info
    name = get_img_name(f, idx)
    boxes = get_img_boxes(f, idx)
    
    # get dimensions of image
    try:
        (h_img, w_img) = cv.imread(path + name).shape[:2]
    except:
        print(f"ERROR: Could not open {name} to get dimensions.")
        print("Make sure image is in same directory as digitStruct.mat")
        print(f"Tried:  {path + name}")
        sys.exit(-3)
        
    # initialize list for annotations
    annots = []
    
    for i in range(len(boxes['label'])):
        # get original bounding values
        (x, y) = (boxes['left'][i], boxes['top'][i])
        (w, h) = (boxes['width'][i], boxes['height'][i])

        # transform x and y
        centerX = x + (w / 2)
        centerY = y + (h / 2)

        # normalize bounding values
        centerX /= w_img
        centerY /= h_img
        w /= w_img
        h /= h_img

        # get label
        label = boxes['label'][i] if boxes['label'][i] != 10 else 0

        # append annotation in Darknet format to annotation list
        annots.append(f'{label} {centerX} {centerY} {w} {h}\n' )
    
    # write annotations to file 
    annot_file = open(path + name.split('.')[0] + '.txt', 'w')
    annot_file.writelines(annots)
    annot_file.close()

def create_annot_files(path):
    if path[-1] != '/':
        path += '/'
    
    try:
        f = h5py.File(f'{path}digitStruct.mat', mode='r')
    except:
        print("ERROR: Could not open file.  Check path to digitStruct.mat")
        sys.exit(-2)
        
    for i in range(len(f['digitStruct/name'])):
        create_annot_file(f, path, i)

def main():
    try:
        path = sys.argv[1]
    except:
        print("ERROR: Must pass path to files as argument.")
        sys.exit(-1)

    create_annot_files(path)

if __name__ == "__main__":
    main()