

import numpy as np
from transformers import AutoTokenizer, __version__, AutoModelForSequenceClassification

import os
import torch


class NLI1Scorer:

    def __init__(self,
                 model = None,
                 batch_size = 64,
                 device = None,
                 direction = 'rh',
                 cross_lingual = False,
                 checkpoint = 0
                 ):
        """

        #labels:
            2: contradiction
            1: neutral
            0: entailment
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        #self.model_name = model.split('/')[1]
        if cross_lingual:
            #model = 'xlm-roberta-base'
            self.model = model
        else:
            model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        self.model_name = model.split('/')[1] if '/' in model else model
        self.checkpoint = checkpoint
        self._tokenizer = get_tokenizer(model)
        self._model = get_model(model, self.device, cross_lingual=cross_lingual, checkpoint=checkpoint)
        self.direction = direction
        self.cross_lingual = cross_lingual

    @property
    def hash(self):
        #return '{}_{}e+{}n+{}c__{}'.format(self.model_name, self.weight_e, self.weight_n, self.weight_c, __version__)
        #return '{}_{}'.format(self.model_name, self.direction)
        return 'crosslingual({}+{})_{}'.format(self.model_name, self.checkpoint, self.direction) if self.cross_lingual else 'monolingual_{}'.format(self.direction)

    def collate_input_features(self, pre, hyp):
        #print(self._tokenizer.model_max_length)
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

        if self.model_name == 'xlm-roberta-large-xnli-anli':
            return probs[:, 0], probs[:, 1], probs[:, 2]

        return probs[:, 2], probs[:, 1], probs[:, 0] #c, n, e
        

def get_tokenizer(model):
    print(model)
    model_dir = 'models/' + model

    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, cache_dir='.cache')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir='.cache')
    return tokenizer

def get_model(model_name, device = None, cross_lingual=False, checkpoint=None):
    print(device)
    '''
    model_dir = 'models/' + model
    if os.path.exists(model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=3, cache_dir='.cache')
    else:

    if not cross_lingual:
        model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=3, cache_dir='.cache')

    else:
    '''
    model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, cache_dir='.cache')

    if cross_lingual and model_name == 'xlm-roberta-base':

        print('>>>')
        if checkpoint == 1:

            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-04-02:19:30_xlm-roberta-base|all|nli/'
                                             'checkpoints/e(2)|i(54000)|snli_test#(0.8767)|xnli_test#(0.6942)|xnli_cross_test#(0.6691)|anli_r1_dev#(0.565)|anli_r2_dev#(0.481)|anli_r3_dev#(0.4458)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 2:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-04-00:19:07_xlm-roberta-base|xnli_cross_all|nli/'
                                             'checkpoints/e(0)|i(2000)|xnli_test#(0.8496)|xnli_cross_test#(0.7982)|anli_r1_dev#(0.531)|anli_r2_dev#(0.418)|anli_r3_dev#(0.4142)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 3:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-04-00:19:07_xlm-roberta-base|xnli_cross_all|nli/'
                                             'checkpoints/e(0)|i(4000)|xnli_test#(0.8433)|xnli_cross_test#(0.7923)|anli_r1_dev#(0.536)|anli_r2_dev#(0.425)|anli_r3_dev#(0.3925)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 4:
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-04-00:19:07_xlm-roberta-base|xnli_cross_all|nli/'
                                             'checkpoints/e(0)|i(12000)|xnli_test#(0.8243)|xnli_cross_test#(0.7802)|anli_r1_dev#(0.549)|anli_r2_dev#(0.438)|anli_r3_dev#(0.4117)/'
                                             'model.pt', map_location=torch.device(device)))
        elif checkpoint == 5:
            #06-06-14:33:18_xlm-r|xnli_cross+mnli_156|nli
            model.load_state_dict(torch.load('../nli_models/anli/saved_models/'
                                             '06-06-14:33:18_xlm-r|xnli_cross+mnli_156|nli/'
                                             'checkpoints/e(1)|i(44000)|xnli_cross_test#(0.6606)|mnli_m_dev#(0.8222)|mnli_mm_dev#(0.8293)/'
                                             'model.pt', map_location=torch.device(device)))

        else:
            raise ValueError('checkpoint not found.')
    model.eval()
    model = model.to(device)
    return model


if __name__ == '__main__':
    # c,n,e

    refs = ["Ich bin in Berlin geboren。", "Ich bin in Hongkong geboren.", "I was born in Taiwan.", "I'm a german",
            "I'm a german"]
    hyps = ['I was born in China.', 'I was born in China', '我来自中国。', "我来自亚洲。",
            "我是欧洲人"]

    #refs = ['我来自柏林。', 'I love animals.','Ich mag Katze.']
    #hyps = ['Ich komme aus Asia.。','我不喜欢动物。','我爱动物，包括猫.']

    #scorer = NLI1Scorer(cross_lingual=True, model="vicgalle/xlm-roberta-large-xnli-anli")
    #scorer = NLI1Scorer(cross_lingual=True, model="xlm-roberta-base")
    #scorer = NLI1Scorer(cross_lingual=True, model="salesken/xlm-roberta-base-finetuned-mnli-cross-lingual-transfer")
    #scorer = NLI1Scorer(cross_lingual=True, model="salesken/xlm-roberta-base-finetuned-mnli-cross-lingual-transfer")
    # scorer = NLI3Scorer(model='bert-mnli')

    #scorer.score(refs, hyps)

