#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
import importlib
importlib.reload(sys)
sys.path.append('../')
#sys.setdefaultencoding('utf8')
import argparse
import os
import time
import rouge
from rouge import Rouge
import torch
from torch.autograd import Variable

from data_util.log import logger
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log, make_html_safe
from train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self.model_path_name = model_name       
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        
        decoded_result = []
        refered_result = []
        article_result = []
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]
            article = batch.original_articles[0]

            #write_for_rouge(original_abstract_sents, decoded_words, counter,
            #                self._rouge_ref_dir, self._rouge_dec_dir)
            decoded_sents = []
            while len(decoded_words) > 0:
                try:
                    fst_period_idx = decoded_words.index(".")
                except ValueError:
                    fst_period_idx = len(decoded_words)
                sent = decoded_words[:fst_period_idx + 1]
                decoded_words = decoded_words[fst_period_idx + 1:]
                decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
            decoded_sents = [make_html_safe(w) for w in decoded_sents]
            reference_sents = [make_html_safe(w) for w in original_abstract_sents]
            decoded_result.append(' '.join(decoded_sents))
            refered_result.append(' '.join(reference_sents))
            article_result.append(article)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        load_file = self.model_path_name
        self.print_original_predicted(decoded_result, refered_result,
                                          article_result, load_file)

        rouge = Rouge() 
        scores = rouge.get_scores(decoded_result, refered_result)
        rouge_1 = sum([x["rouge-1"]["f"] for x in scores]) / len(scores)
        rouge_2 = sum([x["rouge-2"]["f"] for x in scores]) / len(scores)
        rouge_l = sum([x["rouge-l"]["f"] for x in scores]) / len(scores)
        rouge_1_r = sum([x["rouge-1"]["r"] for x in scores]) / len(scores)
        rouge_2_r = sum([x["rouge-2"]["r"] for x in scores]) / len(scores)
        rouge_l_r = sum([x["rouge-l"]["r"] for x in scores]) / len(scores)
        rouge_1_p = sum([x["rouge-1"]["p"] for x in scores]) / len(scores)
        rouge_2_p = sum([x["rouge-2"]["p"] for x in scores]) / len(scores)
        rouge_l_p = sum([x["rouge-l"]["p"] for x in scores]) / len(scores)
        log_str = " rouge_1:" + "%.4f" % rouge_1 + " rouge_2:" + "%.4f" % rouge_2 + " rouge_l:" + "%.4f" % rouge_l
        log_str_r = " rouge_1_r:" + "%.4f" % rouge_1_r + " rouge_2_r:" + "%.4f" % rouge_2_r + " rouge_l_r:" + "%.4f" % rouge_l_r
        logger.info(load_file + " rouge_1:" + "%.4f" % rouge_1 + " rouge_2:" + "%.4f" % rouge_2 + " rouge_l:" + "%.4f" % rouge_l)
        log_str_p = " rouge_1_p:" + "%.4f" % rouge_1_p + " rouge_2_p:" + "%.4f" % rouge_2_p + " rouge_l_p:" + "%.4f" % rouge_l_p
        results_file = os.path.join(self._decode_dir, "ROUGE_results.txt")
        with open(results_file, "w") as f:
            f.write(log_str+'\n')
            f.write(log_str_r + '\n')
            f.write(log_str_p + '\n')


        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)
    
    def print_original_predicted(self, decoded_sents, ref_sents, article_sents,
                                 loadfile):
        filename = "test_" + loadfile.split("_")[1] + ".txt"

        with open(os.path.join(self._rouge_dec_dir, filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: " + article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")


    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_from", type=str, default="0005000.tar")
    opt = parser.parse_args()

    saved_models = os.listdir(config.save_model_path)
    saved_models.sort(key=lambda x: int(x[:-4].split('_')[1]))
    print(saved_models)
    file_idx = saved_models.index(opt.start_from)
    saved_models = saved_models[file_idx:]
    for f in saved_models:
        model_filename = config.save_model_path + '/' + f 
        beam_Search_processor = BeamSearch(model_filename)
        beam_Search_processor.decode()


