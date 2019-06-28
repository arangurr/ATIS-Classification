from argparse import ArgumentParser

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import *


def generate_model(stats, summary=False):

    model = Sequential()
    model.add(Embedding(stats['nb_words'], 64))
    model.add(GRU(100))
    model.add(Dense(stats['nb_classes'], activation='softmax'))

    model.compile('adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if summary:
        model.summary()

    return model


if __name__ == '__main__':

    parser = ArgumentParser('arturo.ai challenge')
    parser.add_argument('--train_path', default='./data/train.iob',
                        help='Path to train.iob file')
    parser.add_argument('--epochs', default=25, type=int,
                        help='Number of epoch to train the model for')
    parser.add_argument('--save', default=True, type=bool,
                        help='Save the best trained model to ./out/model.hdf5')
    parser.add_argument('--plots', default=False, type=bool,
                        help='Show plots with loss and accuracy training history')

    args = parser.parse_args()

    train_s, train_l, train_t = load_data(args.train_path)

    X, y, stats = preprocess(train_s, train_t, pad=True,
                             replace_dig=True, summary=True)

    model = generate_model(stats, summary=True)

    checkpointer = ModelCheckpoint('./out/model.hdf5',
                                   monitor='val_acc',
                                   save_best_only=True)

    hist = model.fit(X, y,
                     validation_split=0.2,
                     epochs=args.epochs,
                     class_weight=stats['weights'],
                     callbacks=[checkpointer] if args.save else [])

    print('Best validation loss: {:.5f} at epoch {}'.format(np.min(hist.history['val_loss']),
                                                            np.argmin(hist.history['val_loss'])))

    print('Best validation accuracy: {:.5f} at epoch {}'.format(np.max(hist.history['val_acc']),
                                                                np.argmax(hist.history['val_acc'])))

    if args.plots:
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss history')
        plt.show()
        plt.savefig('./out/loss.png')

        plt.plot(hist.history['acc'], label='train')
        plt.plot(hist.history['val_acc'], label='validation')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Accuracy history')
        plt.show()
        plt.savefig('./out/acc.png')
