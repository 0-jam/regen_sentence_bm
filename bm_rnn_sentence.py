import argparse
import time
from pathlib import Path
import lzma
import json
from modules.model import BMModel
from modules.plot_result import save_result
import tensorflow as tf
from tensorflow import keras

def load_settings(params_json="settings/default.json"):
    with Path(params_json).open(encoding='utf-8') as params:
        parameters = json.load(params)

    return parameters

def load_test_settings():
    return load_settings("settings/test.json")

## Evaluation methods
# Load learned model
def init_generator(model_dir, text):
    embedding_dim, units, _, cpu_mode = load_settings(model_dir.joinpath("parameters.json")).values()

    generator = BMModel(embedding_dim, units, 1, text, cpu_mode=cpu_mode)
    generator.load(model_dir)

    return generator

def main():
    parser = argparse.ArgumentParser(description="Benchmarking of sentence generation with RNN.")
    parser.add_argument("-c", "--cpu_mode", action='store_true', help="Force to use CPU (default: False)")
    parser.add_argument("-t", "--test_mode", action='store_true', help="Apply settings to train model in short-time for debugging, ignore -e option (default: false)")
    args = parser.parse_args()

    # Retrieve and decompress text
    path = keras.utils.get_file("souseki.txt.xz", "https://drive.google.com/uc?export=download&id=1RnvBPi0GSg07-FhiuHpkwZahGwl4sMb5")
    with lzma.open(path) as file:
        text = file.read().decode()

    if args.test_mode:
        parameters = load_test_settings()

        max_epochs = 2

        gen_size = 100
    else:
        parameters = load_settings()

        if tf.test.is_gpu_available():
            max_epochs = 50
        else:
            max_epochs = 3

        gen_size = 1000

    parameters["cpu_mode"] = args.cpu_mode
    embedding_dim, units, batch_size, cpu_mode = parameters.values()

    ## Create the model
    model = BMModel(embedding_dim, units, batch_size, text, cpu_mode=cpu_mode)
    model.compile()

    today = time.strftime("%Y%m%d")
    result_dir = Path("benchmark_" + today)
    model_dir = result_dir.joinpath("model")
    history = model.fit(model_dir, max_epochs)
    losses = history.history["loss"]

    print("Saving trained model...")
    model.save(model_dir)

    # Generate sentence from the model
    generator = init_generator(model_dir, text)
    generated_text = "".join(generator.generate_text("吾輩は", gen_size))

    # Save results
    print("Saving generated text...")
    with open(str(result_dir) + "/generated_text.txt", 'w', encoding='utf-8') as out:
        out.write(generated_text)

    save_result(losses, save_to=str(result_dir) + "/losses_" + today + ".png")

if __name__ == '__main__':
    main()
