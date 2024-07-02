import pickle
test= pickle.load(open("/home/hx/OpenPCDet/output/waymo_models/pv_rcnn/default/eval/epoch_10/val/default/result.pkl","rb"))
print(len(test))   # 查看数据类型