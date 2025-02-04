from captum.attr import KernelShap, DeepLiftShap, ShapleyValueSampling, Saliency, InputXGradient, IntegratedGradients, LRP, DeepLift
import torch
from sklearn.cluster import KMeans

from rex.rex import rex

baseline = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,])

deeplift_baseline = torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,]])


def shapely_values(model, data, target):
    explainer = ShapleyValueSampling(model)
    res = explainer.attribute(
        data,
        baselines=deeplift_baseline,
        target=torch.tensor([target]),
        n_samples = 5
    )

    return res

def kernel_shap(model, data, target):
    explainer = KernelShap(model.forward)
    explanation = explainer.attribute(
        data, 
        baselines=baseline,
        target=target,
        n_samples = 5
    )

    return explanation

# def deeplift_shap(model, data, target):
#     explainer = DeepLiftShap(model)
#     return explainer.attribute(data[0], baselines=deeplift_baseline, target=target)

def saliency(model, data, target):
    explainer = Saliency(model)

    return explainer.attribute(data, target=target)

def input_x_grad(model, data, target):
    explainer = InputXGradient(model.forward)
    explanation = explainer.attribute(data, target=target)
    return explanation


def integrated_grad(model, data, target):
    explainer = IntegratedGradients(model.forward)
    return explainer.attribute(data, baselines=deeplift_baseline, target=target)


def lrp(model, data, target):
    explainer = LRP(model)
    return explainer.attribute(data, target=target)

def deeplift(model, data, target):
    explainer = DeepLift(model)
    return explainer.attribute(data, baselines=deeplift_baseline, target=target)


def rexplain(model, data, target):
    res = rex(model, data[0])
    return torch.tensor([res])