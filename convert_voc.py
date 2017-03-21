'''Convert VOC PASCAL 2007/2012 xml annotations to a list file.'''

import os
import xml.etree.ElementTree as ET


VOC_LABELS = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)

xml_dir = '/mnt/hgfs/D/download/PASCAL VOC/VOC2007/Annotations/'

f = open('index.txt', 'w')
for xml_name in os.listdir(xml_dir):
    print('converting %s' % xml_name)
    image_name = xml_name[:-4]+'.jpg'
    f.write(image_name+' ')

    tree = ET.parse(os.path.join(xml_dir, xml_name))
    annos = []
    imw = 1.
    imh = 1.
    for child in tree.getroot():
        if child.tag == 'size':
            imw = float(child.find('width').text)
            imh = float(child.find('height').text)

        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = float(bbox.find('xmin').text) / imw
            ymin = float(bbox.find('ymin').text) / imh
            xmax = float(bbox.find('xmax').text) / imw
            ymax = float(bbox.find('ymax').text) / imh
            class_label = VOC_LABELS.index(child.find('name').text)

            annos.append('%.3f %.3f %.3f %.3f %s' % (xmin,ymin,xmax,ymax,class_label))

    f.write('%d %s\n' % (len(annos), ' '.join(annos)))
f.close()
