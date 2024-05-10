import argparse
import torch.utils.data as data
import json
import os
from utility.util import *
from utility.resnest import *
from utility.engine_test_att import *
import numpy as np

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-size', default=224, type=int)
parser.add_argument('-j', '--workers', default=12, type=int)
parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')

class load_data(data.Dataset):
    def __init__(self, attribute, phase='train', num_classes = 0):
        self.phase = phase
        self.attribute = attribute
        self.root = 'data/kfashion_{}'.format(self.attribute)
        self.img_list = []
        self.get_anno()
        self.num_classes = num_classes

    def get_anno(self):
        list_path = 'data/input.json'
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = 'data/category_category_final.json'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        img = Image.open(filename).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        return (img, filename), target

def run_attribute_classifier(model_name ='category'):
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    state = {'batch_size': args.batch_size, 'image_size': args.image_size,
             'evaluate': args.evaluate}

    if model_name == 'category':
        num_classes = 21
        state['resume'] = 'data/model_category_best.pth.tar'

    test_dataset = load_data(attribute = model_name, phase='test', num_classes = num_classes)

    model = resnest50d(pretrained=False, nc=num_classes)
    state['evaluate'] = True
    criterion = nn.MultiLabelSoftMarginLoss()
    engine = Engine(state)
    a = engine.learning(model, criterion, test_dataset, model_name)
    return a

if __name__ == '__main__':
    startTime = time.time()
    print("Starting..")

    lst = []
    p1 = run_attribute_classifier(model_name='category')
    lst.append(p1)
    # print("[DONE] category time spent :{:.4f}".format(time.time() - startTime))
    print(p1)