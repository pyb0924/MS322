import json
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F
import random
import numpy as np

import torch
import tqdm


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def write_result(args, **data):
    data['model'] = args.model
    data['dt'] = datetime.now().isoformat()
    data['batch-size'] = args.batch_size
    data['n_epochs'] = args.n_epochs
    data['dir'] = args.root
    data['problem-type'] = args.type
    data['fold'] = args.fold
    with open('result.json', 'r+', encoding='utf-8') as f:
        prev = json.load(f)
        f.seek(0, 0)
        prev.append(data)
        json.dump(prev, f, ensure_ascii=False, indent=4, sort_keys=True)


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, fold=None,
          num_classes=None, weights=None):
    lr = args.lr
    n_epochs = args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader

        try:
            mean_loss = 0

            for i, (inputs, targets) in enumerate(tl):

                inputs = cuda(inputs)
                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

    if args.ends_flag:
        valid_metrics = validation(model, criterion, valid_loader, num_classes)
        write_result(args, **valid_metrics)
