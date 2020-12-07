from json_to_csv.my_dataloader import load_image, load_mask
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_image_list, val_image_list = get_file_name()

    img_name = str(train_image_list[0])
    mask_name = img_name.replace('images', 'binary_masks')
    mask_name = mask_name.replace('jpg', 'png')
    img = load_image(img_name)
    print(mask_name)
    mask = load_mask(mask_name)

    plt.imshow(img, cmap='gray')
    plt.savefig('img.png')
    plt.imshow(mask, cmap='gray')
    plt.savefig('mask.png')
