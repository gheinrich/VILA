from llava.train import datasets_mixture
import pprint

datasets_mixture.register_datasets_mixtures()
pp = pprint.PrettyPrinter()
pp.pprint("DATASETS")
pp.pprint(datasets_mixture.DATASETS)
pp.pprint("DATASETS_MIXTURES")
pp.pprint(datasets_mixture.DATASETS_MIXTURES)