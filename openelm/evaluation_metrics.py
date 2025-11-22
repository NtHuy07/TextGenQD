import evaluate
import numpy as np

rouge_fn = evaluate.load("rouge")
meteor_fn = evaluate.load("meteor")
chrf_fn = evaluate.load("chrf")
bleu_fn = evaluate.load("bleu")
# bleurt_fn = evaluate.load("bleurt")
scarebleu_fn = evaluate.load("sacrebleu")
bertscore_fn = evaluate.load("bertscore")
perplexity_fn = evaluate.load("perplexity", module_type="metric")

def evaluate_metrics(predictions, references):
    rouge = rouge_fn.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge1 = round(rouge['rouge1'], 4)
    rouge2 = round(rouge['rouge2'], 4)
    rougeL = round(rouge['rougeL'], 4)
    rougeLsum = round(rouge['rougeLsum'], 4)
    meteor = round(meteor_fn.compute(predictions=predictions, references=references)['meteor'], 4)
    chrf = round(chrf_fn.compute(predictions=predictions, references=references)['score'], 4)
    bleu = round(bleu_fn.compute(predictions=predictions, references=references, smooth=True)['bleu'], 4)
    # bleurt = round(np.array(bleurt_fn.compute(predictions=predictions, references=references, smooth=True)['score']).mean(), 4)
    scarebleu = round(np.array(scarebleu_fn.compute(predictions=predictions, references=references)['score']).mean(), 4)
    bertscore = round(bertscore_fn.compute(predictions=predictions, references=references, lang='en')['f1'][0], 4)
    perplexity = round(perplexity_fn.compute(predictions=predictions, model_id="gpt2")['mean_perplexity'], 4)
    
    return {'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'rougeLsum': rougeLsum,
            'meteor': meteor,
            'chrf': chrf,
            'bleu': bleu,
            # 'bleurt': bleurt,
            'scarebleu': scarebleu,
            'bertscore': bertscore,
            'perplexity': perplexity,
            }

def get_metric_names():
    # return ["rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "chrf", "bleu", "bleurt", "scarebleu", "bertscore", "perplexity"]
    return ["rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "chrf", "bleu", "scarebleu", "bertscore", "perplexity"]