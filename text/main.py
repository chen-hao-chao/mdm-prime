import os
import math
import json
import torch
import hydra
import fsspec
import pathlib
import omegaconf
import rich.tree
import rich.syntax
import lightning as L

import dataloader
import diffusion

from utils.logging import get_logger
from utils.file_path import fsspec_exists
from utils.eval_utils import compute_generative_perplexity, compute_entropy
from utils.eval_utils import compute_mauve, compute_self_bleu, compute_ngram_repetition_percentage
from huggingface_hub import hf_hub_download

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  if config.checkpointing.from_huggingface:
    ckpt = hf_hub_download(repo_id="chen-hao-chao/mdm-prime", 
                            filename=config.eval.checkpoint_path,
                            cache_dir=config.data.cache_dir)
    return diffusion.Diffusion.load_from_checkpoint(ckpt, 
                            tokenizer=tokenizer,
                            config=config)
  else:
    return diffusion.Diffusion.load_from_checkpoint(
                            config.eval.checkpoint_path,
                            tokenizer=tokenizer,
                            config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

# ----

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero-Shot Evaluation...')

  logger.info('Loading checkpoint...')
  config.prime.base = math.ceil(config.prime.vocab_size ** (1/config.prime.target_length))
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  trainer = hydra.utils.instantiate(config.trainer,
                                    default_root_dir=os.getcwd(),
                                    callbacks=[],
                                    strategy=hydra.utils.instantiate(config.strategy),
                                    logger=None)

  _, valid_ds, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_test=True, dataset_name=config.data.valid)
      
  if config.eval.compute_zero_shot:
    _, valid_ds_lambada, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True,
      dataset_name='lambada')
    _, valid_ds_wiki, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='wikitext2')
    _, valid_ds_ptb, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='ptb')
    _, valid_ds_lm1b, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='lm1b')
    _, valid_ds_ag_news, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='ag_news')
    _, valid_ds_pubmed, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='scientific_papers_pubmed')
    _, valid_ds_arxiv, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='scientific_papers_arxiv')
    loader_list = [valid_ds, 
                   valid_ds_lambada, 
                   valid_ds_wiki, 
                   valid_ds_ptb, 
                   valid_ds_lm1b, 
                   valid_ds_ag_news, 
                   valid_ds_pubmed, 
                   valid_ds_arxiv]
    trainer.validate(model, loader_list)
  else:
    trainer.validate(model, valid_ds)

def _sample_and_save(config, logger, tokenizer):
  logger.info('Creating directories...')
  p = pathlib.Path(config.sampling.sampling_dir)
  (p / str(config.sampling.split_index)).mkdir(parents=True, exist_ok=True)
  
  logger.info('Loading checkpoint...')
  config.prime.base = math.ceil(config.prime.vocab_size ** (1/config.prime.target_length))
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  if config.sampling.conditions:
    gt_text_list = []
    logger.info("Loading dataset samples...")
    _, _, test_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=True, skip_test=False, 
      dataset_name=config.data.test, valid_seed=0)
    test_loader = iter(test_ds)
  
  logger.info('Generating samples...')
  text_list = []
  accumulative_samples = 0
  num_sample_batches = config.sampling.total_samples // (config.loader.eval_batch_size * config.sampling.variants)
  for i in range(num_sample_batches):
    if i * (config.loader.eval_batch_size*config.sampling.variants) <\
        config.sampling.split_size * config.sampling.split_index:
      continue
    elif i * (config.loader.eval_batch_size*config.sampling.variants) >=\
        config.sampling.split_size * (config.sampling.split_index + 1):
      break

    if config.sampling.conditions:
      conditions = next(test_loader)['input_ids']
      text_condition_samples = model.tokenizer.batch_decode(conditions)
      gt_text_list += text_condition_samples
    else:
      conditions = None
    
    logger.info("Generating the {}-th batch. Accumulative number of samples: {}.".format(i, accumulative_samples))
    with torch.no_grad():
      for j in range(config.sampling.variants):
        samples = model.restore_model_and_sample(num_steps=config.sampling.steps, conditions=conditions)
        text_samples = model.tokenizer.batch_decode(samples)
        text_list += text_samples
        accumulative_samples += len(text_samples)
    
    logger.info('Saving samples...')
    filepath = p / str(config.sampling.split_index) / ('generated_' + str(config.sampling.predictor) + '_' + \
                                                    str(config.sampling.steps) + '_cond=' + \
                                                    str(config.sampling.conditions) + '_nucleus_p=' + \
                                                    str(config.sampling.nucleus_p) + '_seed=' + \
                                                    str(config.seed) + '_' + \
                                                    ('carry_over_' if config.prime.carry_over else "") + \
                                                    ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    with filepath.open("w", encoding ="utf-8") as fp:
      json.dump(text_list, fp)
  
  if config.sampling.conditions:
    logger.info('Saving ground truth for evaluation...')
    gt_filepath = p / str(config.sampling.split_index) / ('gt_' + str(config.sampling.predictor) + '_' + \
                                                      str(config.sampling.steps) + '_cond=' + \
                                                      str(config.sampling.conditions) + '_nucleus_p=' + \
                                                      str(config.sampling.nucleus_p) + '_seed=' + \
                                                      str(config.seed) + '_' + \
                                                      ('carry_over_' if config.prime.carry_over else "") + \
                                                      ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    with gt_filepath.open("w", encoding ="utf-8") as fp:
      json.dump(gt_text_list, fp)
  
  logger.info("Finish sampling!")

import pdb
def _eval_sample(config, logger, tokenizer):
  logger.info("Loading generated samples...")
  number_samples = config.sampling.total_samples
  number_rounds = number_samples // config.sampling.split_size
  p = pathlib.Path(config.sampling.sampling_dir)
  text_list = []
  gt_text_list = []
  if config.performance.load_dataset:
    logger.info("Loading dataset samples...")
    _, _, test_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=True, skip_test=False, 
      dataset_name=config.data.test, valid_seed=0)
    test_loader = iter(test_ds)

  for i in range(number_rounds):
    filepath = p / str(i) / ('generated_' + str(config.sampling.predictor) + '_' + \
                                            str(config.sampling.steps) + '_cond=' + \
                                            str(config.sampling.conditions) + '_nucleus_p=' + \
                                            str(config.sampling.nucleus_p) + '_seed=' + \
                                            str(config.seed) + '_' + \
                                            ('carry_over_' if config.prime.carry_over else "") + \
                                            ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')

    assert filepath.exists(), "Cannot find the generated samples"
    fp = filepath.open("r", encoding ="utf-8")
    text_list_ = json.load(fp)
    text_list += text_list_

    if config.sampling.conditions:
      gt_filepath = p / str(i) / ('gt_' + str(config.sampling.predictor) + '_' + \
                                          str(config.sampling.steps) + '_cond=' + \
                                          str(config.sampling.conditions) + '_nucleus_p=' + \
                                          str(config.sampling.nucleus_p) + '_seed=' + \
                                          str(config.seed) + '_' + \
                                          ('carry_over_' if config.prime.carry_over else "") + \
                                          ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
      
      assert gt_filepath.exists(), "Cannot find the ground truth samples"
      fp = gt_filepath.open("r", encoding ="utf-8")
      gt_text_list_ = json.load(fp)
      gt_text_list += gt_text_list_

    if config.performance.load_dataset:
      samples = next(test_loader)['input_ids']
      text_samples = tokenizer.batch_decode(samples)
      gt_text_list += text_samples
  
  logger.info("Finish loading generated samples! Number of generated samples: {} | true samples: {}".format(len(text_list), len(gt_text_list)))
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda', torch.bfloat16)

  # MAUVE
  if config.performance.mauve:
    logger.info("evaluating mauve...")
    result_path = p / ("stat_" + str(number_samples)) / "mauve"
    file = ('results_mauve_' + str(config.sampling.predictor) + '_' + \
                                str(config.sampling.steps) + '_cond=' + \
                                str(config.sampling.conditions) + '_nucleus_p=' + \
                                str(config.sampling.nucleus_p) + '_' + \
                                ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    result_path.mkdir(parents=True, exist_ok=True)
    mauve_score = compute_mauve(text_list, gt_text_list, 
                                logger=logger, 
                                mauve_scaling_factor=config.performance.mauve_scaling_factor)
    logger.info('mauve: {}'.format(mauve_score.mauve))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(mauve_score.mauve), fp)
  
  # Self-BLEU
  if config.performance.bleu:
    logger.info("evaluating self-bleu...")
    result_path = p / ("stat_" + str(number_samples)) / "self_bleu"
    file = ('results_self_bleu_' + str(config.sampling.predictor) + '_' + \
                                str(config.sampling.steps) + '_cond=' + \
                                str(config.sampling.conditions) + '_nucleus_p=' + \
                                str(config.sampling.nucleus_p) + '_' + \
                                ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    result_path.mkdir(parents=True, exist_ok=True)
    self_bleu = compute_self_bleu(model, text_list, logger=logger)
    logger.info('self_bleu (generated text): {}'.format(self_bleu))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(self_bleu), fp)

  if config.performance.bleu_dataset:
    logger.info("evaluating self-bleu (dataset)...")
    result_path = p / ("stat_" + str(number_samples)) / "self_bleu" / 'results_self_bleu_dataset.json'
    self_bleu_dataset = compute_self_bleu(model, gt_text_list, logger=logger)
    logger.info('self_bleu (dataset): {}'.format(self_bleu_dataset))
    with result_path.open("w", encoding ="utf-8") as fp:
      json.dump(str(self_bleu_dataset), fp)
      
  # Repetition
  if config.performance.repetition:
    logger.info("evaluating repetition...")
    result_path = p / ("stat_" + str(number_samples)) / "repetition" 
    file = ('results_repetition_' + str(config.sampling.predictor) + '_' + \
                                str(config.sampling.steps) + '_cond=' + \
                                str(config.sampling.conditions) + '_nucleus_p=' + \
                                str(config.sampling.nucleus_p) + '_' + \
                                ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    result_path.mkdir(parents=True, exist_ok=True)
    repetition = compute_ngram_repetition_percentage(text_list)
    logger.info('repetition (generated text): {:.2f} %'.format(repetition))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(repetition), fp)

  if config.performance.repetition_dataset:
    logger.info("evaluating repetition (dataset)...")
    result_path = p / ("stat_" + str(number_samples)) / "repetition" / 'results_repetition_dataset.json'
    repetition_dataset = compute_ngram_repetition_percentage(gt_text_list)
    logger.info('repetition (dataset): {:.2f} %'.format(repetition_dataset))
    with result_path.open("w", encoding ="utf-8") as fp:
      json.dump(str(repetition_dataset), fp)

  # Entropy
  if config.performance.entropy:
    logger.info("evaluating entropy...")
    result_path = p / ("stat_" + str(number_samples)) / "entropy"
    file = ('results_entropy_' + str(config.sampling.predictor) + '_' + \
                                str(config.sampling.steps) + '_cond=' + \
                                str(config.sampling.conditions) + '_nucleus_p=' + \
                                str(config.sampling.nucleus_p) + '_' + \
                                ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    result_path.mkdir(parents=True, exist_ok=True)
    entropy = compute_entropy(model, text_list, logger=logger)
    logger.info('entropy (generated text): {}'.format(entropy))

    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(entropy), fp)

  if config.performance.entropy_dataset:
    logger.info("evaluating entropy (dataset)...")
    result_path = p / ("stat_" + str(number_samples)) / "entropy" 
    file = 'results_entropy_dataset.json'
    entropy_dataset = compute_entropy(model, gt_text_list, logger=logger)
    result_path.mkdir(parents=True, exist_ok=True)
    logger.info('entropy (dataset): {}'.format(entropy_dataset))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(entropy_dataset), fp)
  
  # Generative Perplexity
  if config.performance.gen_perplexity:
    logger.info("evaluating generative perplexity...")
    os.environ["TRANSFORMERS_CACHE"] = config.data.cache_dir
    result_path = p / ("stat_" + str(number_samples)) / "gen_ppl"
    file = ('results_gen_ppl_' + str(config.sampling.predictor) + '_' + \
                                str(config.sampling.steps) + '_cond=' + \
                                str(config.sampling.conditions) + '_nucleus_p=' + \
                                str(config.sampling.nucleus_p) + '_' + \
                                ('fp64' if config.sampling.sampling_dtype == 'fp64' else 'fp32') + '.json')
    result_path.mkdir(parents=True, exist_ok=True)
    gen_ppl = compute_generative_perplexity(model, text_list, logger=logger)
    logger.info('gen_ppl (generated text): {}'.format(gen_ppl))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(gen_ppl), fp)

  if config.performance.gen_perplexity_dataset:
    logger.info("evaluating generative perplexity (dataset)...")
    model.gen_ppl_metric.reset()
    result_path = p / ("stat_" + str(number_samples)) / "gen_ppl" 
    file = 'results_gen_ppl_dataset.json'
    gen_ppl_dataset = compute_generative_perplexity(model, gt_text_list, logger=logger)
    result_path.mkdir(parents=True, exist_ok=True)
    logger.info('gen_ppl (dataset): {}'.format(gen_ppl_dataset))
    with (result_path / file).open("w", encoding ="utf-8") as fp:
      json.dump(str(gen_ppl_dataset), fp)

  logger.info("Groud Truth Length: {}".format(len(gt_text_list)))
  logger.info("Generated Sample Length: {}".format(len(text_list)))

def _train(config, logger, tokenizer):
  logger.info('Initiating callbacks and wandb...')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  logger.info('Constructing the model...')
  config.prime.base = math.ceil(config.prime.vocab_size ** (1/config.prime.target_length))
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and fsspec_exists(config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None
  model = diffusion.Diffusion(config, tokenizer=tokenizer)

  logger.info('Loading datasets...')
  train_ds, _, _ = dataloader.get_dataloaders(
    config, tokenizer, skip_valid=True, skip_test=True, 
    dataset_name=config.data.train)
  _, valid_ds, _ = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, skip_test=True, 
    dataset_name=config.data.valid)
  _print_batch(train_ds, valid_ds, tokenizer)

  _, _, test_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, skip_valid=True, skip_test=False, 
    dataset_name=config.data.test, valid_seed=config.seed)
  model.test_eval_loader = test_ds

  logger.info('Start Training...')
  trainer = hydra.utils.instantiate(config.trainer,
                                    default_root_dir=os.getcwd(),
                                    callbacks=callbacks,
                                    strategy=hydra.utils.instantiate(config.strategy),
                                    logger=wandb_logger)
  if config.eval.compute_zero_shot:
    _, valid_ds_lambada, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True,
      dataset_name='lambada')
    _, valid_ds_wiki, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='wikitext2')
    _, valid_ds_ptb, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='ptb')
    _, valid_ds_lm1b, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='lm1b')
    _, valid_ds_ag_news, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='ag_news')
    _, valid_ds_pubmed, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='scientific_papers_pubmed')
    _, valid_ds_arxiv, _ = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, skip_valid=False, skip_test=True, 
      dataset_name='scientific_papers_arxiv')
    loader_list = [valid_ds, 
                   valid_ds_lambada, 
                   valid_ds_wiki, 
                   valid_ds_ptb, 
                   valid_ds_lm1b, 
                   valid_ds_ag_news, 
                   valid_ds_pubmed, 
                   valid_ds_arxiv]
    trainer.fit(model, train_dataloaders=train_ds, 
                       val_dataloaders=loader_list, 
                       ckpt_path=ckpt_path)
  else:
    trainer.fit(model, train_dataloaders=train_ds, 
                       val_dataloaders=valid_ds, 
                       ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'sample_and_save':
    _sample_and_save(config, logger, tokenizer)
  elif config.mode == 'sample_quality_eval':
    _eval_sample(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()