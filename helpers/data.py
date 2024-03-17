import json
import torch
from mmcv.parallel import DataContainer as DC

def create_data(img_pre, project_matrix):
# def create_data():

    # cuda0 = torch.device('cuda:0')
    
    f = open('data.json')
    data_json = json.load(f)

    data = dict()
    import pdb; pdb.set_trace()
    # data['img_metas']._data[0].shape
    data['img_metas'] = dict({'ori_shape': [torch.tensor([1280]), torch.tensor([1920])]})  
    
    # data['img']._data[0].shape --> torch.Size([16, 3, 360, 480])
    img = torch.tensor(data_json['img'])
    img[0, :, :, :] = img_pre
    data['img'] = DC([img])

    # N - number of anchors
    # C - ? TODO: figure out
    # data['gt_3dlanes']._data[0][0].shape --> torch.Size([N, C])
    data['gt_3dlanes'] = DC([torch.tensor(data_json['gt_3dlanes'])])

    # data['gt_project_matrix']._data[0].shape --> torch.Size([16, 1, 3, 4])
    # gt_project_matrix = torch.tensor(data_json['gt_project_matrix'])
    gt_project_matrix = torch.tensor(project_matrix) # [1 1 3 4]
    data['gt_project_matrix'] = DC([gt_project_matrix])

    # data['mask']._data[0].shape --> torch.Size([16, 1, 360, 480])
    data['mask'] = DC([torch.tensor(data_json['mask'])])

    return data

def save_DC_in_json(data):

    data_json = {}

    for key in data.keys():
        if key == 'img':
            data_json[key] = data['img']._data[0].cpu().numpy().tolist()
        elif key == 'gt_3dlanes':
            data_json[key] = [data['gt_3dlanes']._data[0][0].cpu().numpy().tolist()]
        elif key == 'gt_project_matrix':
            data_json[key] = data['gt_project_matrix']._data[0].cpu().numpy().tolist()
        elif key == 'mask':
            data_json[key] = data['mask']._data[0].cpu().numpy().tolist()

    with open('data.json', 'w') as f:
        json.dump(data_json, f)


if __name__ == '__main__':
    create_data()