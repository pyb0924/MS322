from dataset import load_image, load_all_mask
import torch
from pathlib import Path
from albumentations.pytorch.functional import img_to_tensor
from models.models import UNet11, UNet16, UNet, LinkNet34, TDSNet
import matplotlib.pyplot as plt
import cv2

moddel_list = {
    'UNet11': UNet11,
    'UNet16': UNet16,
    'UNet': UNet,
    'LinkNet34': LinkNet34,
    #'TDSNet': TDSNet
}
type_list = ['binary', 'parts', 'instruments']

img_file_name = 'cropped_train/instrument_dataset_8/images/frame000.jpg'

if __name__ == '__main__':

    image = load_image(img_file_name)

    mask_binary, mask_parts, mask_instruments = load_all_mask(img_file_name)

    image = img_to_tensor(image)

    image = torch.unsqueeze(image, 0)
    mask = {'mask_binary': mask_binary, 'mask_parts': mask_parts, 'mask_instruments': mask_instruments}

    fig, axes = plt.subplots(3, 6)
    plt.subplots_adjust(wspace=0, hspace=0)

    for i, problem_type in enumerate(type_list):
        ax = axes[i, 0]
        ax.imshow(mask['mask_' + problem_type], cmap='gray')
        ax.axis('off')

    for i, moddel in enumerate(moddel_list):
        if not moddel == 'TDSNet':
            for j, problem_type in enumerate(type_list):

                if problem_type == 'parts':
                    num_classes = 4
                elif problem_type == 'binary':
                    num_classes = 1
                else:
                    num_classes = 8
                model_name = moddel_list[moddel]
                model = model_name(num_classes=num_classes)
                model_path = Path('runs') / '{model}_{type}'.format(model=moddel, type=problem_type) / 'model_3.pt'
                print(model_path)
                if model_path.exists():
                    state = torch.load(str(model_path))
                    model.load_state_dict(state['model'], strict=False)

                with torch.no_grad():
                    #model.train()
                    model.eval()
                    output = model(image)
                    if problem_type == 'binary':
                        output = (output > 0).float()
                        output = output.data.detach().numpy()
                        output = output[0]
                    else:
                        output = output.data.detach().numpy().argmax(axis=1)

                ax = axes[j, i + 1]
                ax.imshow(output[0], cmap='gray')
                ax.axis('off')

        else:
            model = TDSNet(num_classes=8)
            model_path = Path('runs') / 'TDSNet' / 'model_{0}.pt'
            if model_path.exists():
                state = torch.load(str(model_path))
                model.load_state_dict(state['model'])

                with torch.no_grad():
                    model.eval()
                    output_binary, output_parts, output_instruments = model(image)

                    output_binary = torch.squeeze(output_binary).detach().numpy()
                    output_parts = torch.squeeze(output_parts).detach().numpy()
                    output_instruments = torch.squeeze(output_instruments).detach().numpy()

                ax = axes[0, i + 1]
                ax.imshow(output_binary, cmap='gray')
                ax.axis('off')
                ax = axes[1, i + 1]
                ax.imshow(output_parts, cmap='gray')
                ax.axis('off')
                ax = axes[2, i + 1]
                ax.imshow(output_instruments, cmap='gray')
                ax.axis('off')
    plt.show()
