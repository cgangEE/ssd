#!/usr/bin/env python

from symbol.symbol_factory_mobi import get_symbol_train
import mxnet as mx

def gao():
    net = get_symbol_train('mobilenetTmp', 300, 
            num_classes=4, nms_thresh=0.5, 
            force_suppres=True, nms_topk=400)
    data_shape = (1, 3, 300, 300)
    label_shape = (1, 1, 5)
    mx.viz.plot_network(net, shape={"data":data_shape, "label":label_shape},
            node_attrs={"hide_weights":"true", 
            "fixedsize":'false', "shape":'oval'} ).view()

if __name__ == '__main__':
    gao()

