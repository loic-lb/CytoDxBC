import torch
import timm

def create_cytoFM_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Identity()
    ck = torch.load("./cytofm_weights.pth", map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict())
    for key, value in ck['teacher'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
        elif key.startswith('module.backbone'):
            output_dict['state_dict'][key[16:]] = value
    model.load_state_dict(output_dict['state_dict'], strict=True)
    return model