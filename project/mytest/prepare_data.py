from pathlib import Path

original_path = Path('../raw_data')
des_path = Path('../train_data')

folder = 'instrument_dataset_'

if __name__ == '__main__':
    for index in range(1, 9):
        instrument_folder = des_path / (folder + str(index))
        instrument_folder.exists()

        image_folder = instrument_folder / 'images'
        image_folder.mkdir(exist_ok=True, parents=True)
        binary_mask_folder = instrument_folder / 'binary_masks'
        binary_mask_folder.mkdir(exist_ok=True, parents=True)
        parts_mask_folder = instrument_folder / 'parts_masks'
        parts_mask_folder.mkdir(exist_ok=True, parents=True)
        instrument_mask_folder = instrument_folder / 'instruments_masks'
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)
