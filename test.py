from keras.models import load_model
from utils import *
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser('arturo.ai challenge')
    parser.add_argument('--test_path', default='./data/test.iob',
                        help='Path to test.iob file')
    parser.add_argument('--model', default='./out/model.hdf5',
                        help='Path to model.hdf5 file')
    parser.add_argument('--tokenizer', default='./out/tokenizer.pickle',
                        help='Path to tokenizer.pickle file')
    parser.add_argument('--labelizer', default='./out/label_encoder.pickle',
                        help='Path to label_encoder.pickle file')

    args = parser.parse_args()

    test_s, test_l, test_t = load_data(args.test_path)

    with open(args.labelizer, 'rb') as le:
        labelizer = pickle.load(le)

    with open(args.tokenizer, 'rb') as tkn:
        tokenizer = pickle.load(tkn)

    X, y, _ = preprocess(test_s, test_t, pad=True,
                             labelizer=labelizer,
                             tokenizer=tokenizer)

    model = load_model(args.model)

    loss, acc = model.evaluate(X, y)
    print(f'loss: {loss}\naccuracy: {acc}')
