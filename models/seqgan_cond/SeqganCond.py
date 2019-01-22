import json
from time import time

from models.Gan import Gan
from models.seqgan_cond.SeqganCondDataLoader import DataLoader, DisDataloader
from models.seqgan_cond.SeqganCondDiscriminator import Discriminator
from models.seqgan_cond.SeqganCondGenerator import Generator
from models.seqgan_cond.SeqganCondReward import Reward
from utils.metrics.Cfg import Cfg
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleCfg import OracleCfg
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *


class SeqganCond(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0
        self.safe = False # Just for debugging purposes turned off

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):

        ## Generate samples (again...)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)

        ## We load the training data
        ## Remember: The oracle file is just our dataset encoded with integers!
        ## In the load_train_data the data is shuffled, too.
        ## Thus it needs to be called again and again to ensure, that the discrimnator gets trained with other data every time
        ## The generator file contains the data which has been generated by the generator (obviously...)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)

        # Now we do this three times...
        for _ in range(3):
            self.dis_data_loader.next_batch() # This is random bullshit, I guess
            x_batch, y_batch = self.dis_data_loader.next_batch() # Now we assign x_batch and y_batch
            # x_batch contains the training examples
            # y_batch contains the labels, i.e. [0, 1] for training data and [1,0] for generator data
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            # And the discriminator is trained now
            loss,_ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)
            print(loss)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            # It is none in other case, so nvm.
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:

                ### HEADER OF CSV
                self.log.write('epochs, ') # This might be the most crazy way to generate a csv file...
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',') # Write metricname
                self.log.write('\n') # Write newline
                ### END OF HEADER

                ## The super method prints the metrices to the console
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        ## This gets called, if logging is disabled
        ## The super method prints the metrices to the console
        return super().evaluate()

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-SeqganCond.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-SeqganCond-cfg.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)

        #Notice: No real data stuff is done here, just general initializiation...
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator) # Just a normal setter method, possibly as an easier interface to extension or so

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator) # Just a normal setter method, possibly as an easier interface to extension or so

        # Here is where the stuff gets interesting, because we start to load the data
        # Difference between both is that the gen_dataloader loads only the "data" file,
        # while the dis_dataloader loads positive and negative files
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None ## There is no oracle yet
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        # Just a normal setter method, possibly as an easier interface to extension or so
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        tokens = get_tokenlized(data_loc) # Get the tokenized sequences
        word_set = get_word_list(tokens) # Get the set of words (Notice: This has been done in text_precess before, so its pretty inefficient)

        [word_index_dict, index_word_dict] = get_dict(word_set) # Gets the word->Index and Index-> Word mapping for the words
        with open(self.oracle_file, 'w') as outfile:
            # Now we create an oracle file
            # This oracle file contains all our sequences encoded to integers
            # If the sentence is not long enough, i.e. < sequence length, it gets filled up with EOF characters
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict # We return the word->Index and Index->Word Mapping

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim) # Just add docsim to the metrics list which is evaluated later

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll) # Just add nll to the metrics list


    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc) # Contains word->Index, index->Word Dictionary
        self.init_real_metric() # Add the DocEmbSim and the NLL-Test to the metrices

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r', encoding="utf-8") as file:
                codes = get_tokenlized(self.generator_file) # Notice: The generator-file is a list of codes, not texts --> we get codes
            with open(self.test_file, 'w', encoding="utf-8") as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict)) # We write the codes back as text. --> So we write the generator's output (which were codes before) back as text

        ### Here is where the training starts.
        saver = tf.train.Saver()

        # After variables have been declared in tensorflow, they have to be initialized
        # The variables have been declared in the init_training method in which the generator and the discriminator have been set.
        self.sess.run(tf.global_variables_initializer())

        # Just some parameters, randomly posted inside a function where no one will ever find them and wonders why changing parameters does not work
        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-SeqganCond-real.csv', 'w') # In this file the metrice's output will be saved

        #### PRE-TRAIING STUFF

        # First we generate samples by using the generator to write them into the generator file.
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)

        # Then we generate batches using the oracle file.
        # Remember: In init_real_training we wrote the content of our training dataset into the oracle file encoded with integers
        # So, we are using the dataset as input (more or less)
        # --> Verified
        self.gen_data_loader.create_batches(self.oracle_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()

            # So here we do an pre training epoch
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)

            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                # We generate new samples
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)

                # This method has been defined above:
                # In this case its function is to turn the codes from the generated samples back to the words
                get_real_test_file()

                # Calculate, Print and Save the metrices
                self.evaluate()
                if self.safe:

                    # Save model
                    save_path = saver.save(self.sess, "/home/hilko/saves/pre_train_gen_{}".format(str(epoch)))

        print('start pre-train discriminator:')

        # Does nothing, just returns...
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))

            ## --> See method above
            self.train_discriminator()

            # Save sometimes...
            if epoch % 5 == 0:
                if self.safe:
                    save_path = saver.save(self.sess, "/home/hilko/saves/pre_train_disc_{}".format(str(epoch)))


        # Does nothing, just returns...
        self.reset_epoch()

        # Most complex part: Train Generator
        print('adversarial training:')

        # Hardcoded update_rate for reward smileyface
        # generator must be lstm, according to Reward class

        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):


                # The generator generates samples
                samples = self.generator.generate(self.sess)

                # Then we calculate the rewards
                # 16 = rollout_num
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                # --> Put it into the session again.
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                # Every 5 epochs we generate new samples which are saved into a file
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

                # Save sometimes...
                if self.safe:
                    save_path = saver.save(self.sess, "/home/hilko/saves/adv_train_{}".format(str(epoch)))

            # And the weights for the reward are updates
            self.reward.update_params()
            for _ in range(15):
                # Discriminator is trained again
                self.train_discriminator()
