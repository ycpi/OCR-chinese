# coding:utf-8
from config import ocrModel, LSTMFLAG, GPU
from torch.autograd import Variable
from config import chinsesModel
from collections import OrderedDict
from crnn import keys
from crnn.models.crnn import CRNN
from crnn import dataset
from crnn import util
import torch.utils.data
import torch
import sys
sys.path.insert(1, "./crnn")


class crnn(object):
    def __init__(self):
        if chinsesModel:
            alphabet = keys.alphabetChinese
        else:
            alphabet = keys.alphabetEnglish

        self.converter = util.strLabelConverter(alphabet)
        if torch.cuda.is_available() and GPU:
            # LSTMFLAG=True crnn 否则 dense ocr
            self.model = CRNN(32, 1, len(alphabet)+1, 256,
                              1, lstmFlag=LSTMFLAG).cuda()
        else:
            self.model = CRNN(32, 1, len(alphabet)+1, 256,
                              1, lstmFlag=LSTMFLAG).cpu()

        state_dict = torch.load(
            ocrModel, map_location=lambda storage, loc: storage)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        # load params

        self.model.load_state_dict(new_state_dict)

    def crnnOcr(self, image):
        scale = image.size[1]*1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        # print "im size:{},{}".format(image.size,w)
        transformer = dataset.resizeNormalize((w, 32))
        if torch.cuda.is_available() and GPU:
            image = transformer(image).cuda()
        else:
            image = transformer(image).cpu()

        image = image.view(1, *image.size())
        image = Variable(image)
        self.model.eval()
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

        return sim_pred