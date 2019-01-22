import numpy as np
import tensorflow as tf


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    #outputfile = generatorfile ==> codes, not strings!
    # Generate Samples and saves them to file (if wanted)
    # Additionally returns a string with all the codestrings, separated by newline \n
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []

    # Reset pointer is required to start with the first batch again
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()

        # And now we do the pretrrain step, i.e. we do one batch iteration
        # --> Look at SeqganCondGenerator
        _, g_loss = trainable_model.pretrain_step(sess, batch)

        # And we apply the loss to our losses array
        supervised_g_losses.append(g_loss)

    # The mean of all losses is returned.
    # Think about: Why mean, not median?
    return np.mean(supervised_g_losses)
