from Experiments.Baselines.ResnetClassification import main, main2
from Experiments.Baselines.DeepEnsemble import test1
from Experiments.Baselines.PackedEnsemble_test import PEResnet18_classification, PEResnet50_classification
from Experiments.DiversityTests.Diversity import diversity_test


if __name__ == "__main__":
    #main2()
    test1()
    diversity_test()
    #main()
    #PEResnet18_classification()
    #PEResnet50_classification()