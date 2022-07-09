

import numpy as np
from transformers import AutoTokenizer, __version__, AutoModelForSequenceClassification

import os
import torch


class NLI2Scorer:

    def __init__(self,
                 model = None,
                 batch_size = 64,
                 device = None,
                 direction = 'rh',
                 cross_lingual = False,
                 checkpoint = 0
                 ):
        """
        Args:
        :param metric:
        :param model:
        :param num_layer:
        :param btach_size:
        :param nthreads:
        :param idf:
        :param device:
        :param strategy:
        #labels:
            0: contradiction
            1: neutral
            2: entailment
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        if cross_lingual:
            #model = 'xlm-roberta-base'
            self.model = model
        else:
            model = "microsoft/deberta-large-mnli"

        self.model_name = model.split('/')[1] if '/' in model else model

        self.checkpoint = checkpoint
        self._tokenizer = get_tokenizer(model)
        self._model = get_model(model, self.device, cross_lingual=cross_lingual, checkpoint=checkpoint)
        self.direction = direction
        self.cross_lingual = cross_lingual

    @property
    def hash(self):
        return 'crosslingual({}+{})_{}'.format(self.model_name, self.checkpoint,
                                               self.direction) if self.cross_lingual else 'monolingual_{}'.format(
            self.direction)

    def collate_input_features(self, pre, hyp):
        tokenized_input_seq_pair = self._tokenizer.encode_plus(pre, hyp,
                                                         max_length=self._tokenizer.model_max_length,
                                                         return_token_type_ids=True, truncation=True)



        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)


        return input_ids, token_type_ids, attention_mask

    def score(self, refs, hyps):

        probs = []

        with torch.no_grad():
            for ref, hyp in zip(refs, hyps):
                if self.direction == 'rh':
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(ref, hyp)
                else:
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(hyp, ref)

                logits = self._model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)[0]
                prob = torch.softmax(logits, 1).detach().cpu().numpy()
                probs.append(prob)


        #print(probs)
        probs = np.concatenate(probs, 0)
        #print(probs)

        #return probs[:,2], probs[:,1], probs[:,0] #c, n, e
        if self.cross_lingual:
            return probs[:, 2], probs[:, 1], probs[:, 0]  # c, n, e # it seems the order is the same. remove this line?????
        #else:
        return probs[:, 0], probs[:, 1], probs[:, 2]  #c, n, e

def get_tokenizer(model):
    model_dir = 'models/' + model

    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, cache_dir='.cache')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir='.cache')
    return tokenizer

def get_model(model_name, device = 'cuda', cross_lingual=False, checkpoint=0):
    '''
    model_dir = 'models/' + model
    if os.path.exists(model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=3, cache_dir='.cache')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=3)

    '''
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, cache_dir='.cache')

    if cross_lingual and model_name == 'microsoft/mdeberta-v3-base':
        if checkpoint == 1:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-06-12:51:59_mdeberta|xnli_cross+mnli|nli/'
                                             'checkpoints/e(1)|i(46000)|xnli_cross_test#(0.7267)|mnli_m_dev#(0.8664)|mnli_mm_dev#(0.8676)/'
                                             'model.pt',map_location=torch.device(device)))
        elif checkpoint == 2:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-06-12:51:59_mdeberta|xnli_cross+mnli|nli/'
                                             'checkpoints/e(1)|i(54000)|xnli_cross_test#(0.7228)|mnli_m_dev#(0.8731)|mnli_mm_dev#(0.8735)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 3:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-06-14:14:14_mdeberta|xnli_cross+mnli_256|nli/'
                                             'checkpoints/e(1)|i(48000)|xnli_cross_test#(0.7288)|mnli_m_dev#(0.8688)|mnli_mm_dev#(0.8698)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 4:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-06-14:14:14_mdeberta|xnli_cross+mnli_256|nli/'
                                             'checkpoints/e(1)|i(54000)|xnli_cross_test#(0.727)|mnli_m_dev#(0.8765)|mnli_mm_dev#(0.8743)/'
                                             'model.pt', map_location=torch.device(device)))

        else:
            raise ValueError('checkpoint not found.')
    model.eval()
    model = model.to(device)
    return model

if __name__ == '__main__':
    # c,n,e

    refs = ["I come from Berlin", "Ich war in Berlin geboren。", "Ich war in Beijing geboren.", "I'm not from China.", "I'm a german",
            "I'm a german"]
    hyps = ["I am from Berlin", 'I was born in China.', '我出生于中国。', '我来自中国。', "我是亚洲人。",
            "我是欧洲人"]

    #refs = ['我来自柏林。', 'I love animals.','Ich mag Katze.']
    #hyps = ['Ich komme aus Asia.。','我不喜欢动物。','我爱动物，包括猫.']

    scorer = NLI2Scorer()
    scorer.score(refs,hyps)
    # e, n, c
    scorer = NLI2Scorer(cross_lingual=True, model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    scorer.score(refs, hyps)

    scorer = NLI2Scorer(cross_lingual=True, model="microsoft/mdeberta-v3-base", checkpoint=1)
    scorer.score(refs, hyps)
    scorer = NLI2Scorer(cross_lingual=True, model="microsoft/mdeberta-v3-base", checkpoint=2)
    scorer.score(refs, hyps)
    scorer = NLI2Scorer(cross_lingual=True, model="microsoft/mdeberta-v3-base", checkpoint=3)
    scorer.score(refs, hyps)
    scorer = NLI2Scorer(cross_lingual=True, model="microsoft/mdeberta-v3-base", checkpoint=4)
    scorer.score(refs, hyps)
    # scorer = NLI3Scorer(model='bert-mnli')

    #scorer.score(refs, hyps)





