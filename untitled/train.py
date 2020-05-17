import numpy as np
import pandas as pd
import tensorflow as tf
import os, json
from data_process import build_vocab_morphs, sentence_to_index_morphs, batch_iter
from word2vec import make_embedding_vectors
from model import CNN

if __name__ == "__main__":
    DIR = "models"

    # build dataset
    train = pd.read_csv('./data/train.txt', delimiter='\t')
    test = pd.read_csv('./data/test.txt', delimiter='\t')
    data = train.append(test)
    x_input = data.document
    y_input = data.label
    max_length = 30
    print('데이터로부터 정보를 얻는 중입니다.')
    embedding, vocab, vocab_size = make_embedding_vectors(list(x_input))
    print('완료되었습니다.')

    # save vocab, vocab_size, max_length
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    # save configuration
    with open('config.txt', 'w') as f:
        f.write(str(vocab_size) + '\n')
        f.write(str(max_length))

    # No open session in tf 2.0
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus, True)
        except RuntimeError as e:
            print(e)

    # make model instance
    model = CNN(vocab_size=vocab_size, sequence_length=max_length, testmode=False) #, trainable=True)

    # assign pretrained embedding vectors
    # model.embedding_assign(embedding)

    # make train batches
    batches = batch_iter(list(zip(x_input, y_input)), batch_size=64, num_epochs=5)

    # model saver
    saver = tf.train.Checkpoint(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    print('모델 훈련을 시작합니다.')
    avgLoss = []
    for step, batch in enumerate(batches):
        x_train, y_train = zip(*batch)
        x_train = sentence_to_index_morphs(x_train, vocab, max_length)
        # l, _ = model.train(x_train, y_train)
        model.fit(x_train, y_train)
        # avgLoss.append(l)
        # if step % 200 == 0:
        #     print('batch:', '%04d' % step, 'loss:', '%05f' % np.mean(avgLoss))
        saver.save(os.path.join(DIR, "model")) #, global_step=step)
        # avgLoss = []