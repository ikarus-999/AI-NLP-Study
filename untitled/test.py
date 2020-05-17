from model import CNN
from data_process import sentence_to_index_morphs
import tensorflow as tf
import re, json

if __name__ == "__main__":
    PATH = "models"

    # load vocab, vocab_size, max_length
    with open('vocab.json', 'r') as fp:
        vocab = json.load(fp)

    # load configuration
    with open('config.txt', 'r') as f:
        vocab_size = int(re.sub('\n', '', f.readline()))
        max_length = int(f.readline())

    # No open session in tf2.0
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
    model = CNN(vocab_size=vocab_size, sequence_length=max_length, testmode=True) #, trainable=True)

    # load trained model
    saver = tf.train.Checkpoint()
    saver.restore(tf.train.latest_checkpoint(PATH))

    # inference
    while True:
        test = input("User >> ")
        if test == "exit":
            break
        speak = sentence_to_index_morphs([test], vocab, max_length)

        label, prob = model(speak)

        if prob[0] < 0.6:
            response = '차분해 보이시네요 :)'
        else:
            if label[0] == 0:
                response = '기분이 좋지 않아 보여요 :('
            else:
                response = '기분이 좋아 보이시네요!'

        print("Bot >> ", response, "긍부정여부: ", label[0]," 확률: ", prob[0], "\n")