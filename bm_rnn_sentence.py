# Regulation ver: 20190319
import time
from pathlib import Path

from modules.plot_result import save_result
from modules.bmmodel import BMModel


def main():
    model = BMModel()

    today = time.strftime('%Y%m%d')
    save_dir = Path('benchmark_' + today)
    model_save_dir = save_dir.joinpath('model')

    history, result_text = model.fit()
    print('Saving trainer ...')
    model.save_trainer(model_save_dir)
    losses = history.history['loss']

    print('Generating result graph ...')
    save_result(losses, save_to=str(save_dir) + '/losses_' + today + '.png')

    # Save results
    print('Generating text:')
    model.build_generator(model_save_dir)
    generated_text = ''.join(model.generate_text('吾輩は', 1000))

    print('Saving generator ...')
    model.save_generator(model_save_dir)

    print('Saving generated text...')
    with Path(str(save_dir) + '/generated_text.txt').open('w', encoding='utf-8') as out:
        out.write(generated_text + '\n' + result_text)


if __name__ == '__main__':
    main()
