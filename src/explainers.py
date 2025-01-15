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

    return torch.softmax(res, dim=1)

def kernel_shap(model, data, target):
    explainer = KernelShap(model.forward)
    explanation = explainer.attribute(
        data, 
        baselines=baseline,
        target=target,
        n_samples = 5
    )

    return torch.softmax(explanation, dim=1)

# def deeplift_shap(model, data, target):
#     explainer = DeepLiftShap(model)
#     return explainer.attribute(data[0], baselines=deeplift_baseline, target=target)

def saliency(model, data, target):
    explainer = Saliency(model)

    return explainer.attribute(data, target=target)

def input_x_grad(model, data, target):
    explainer = InputXGradient(model.forward)
    explanation = explainer.attribute(data, target=target)
    return torch.softmax(explanation, dim=1)


def integrated_grad(model, data, target):
    explainer = IntegratedGradients(model.forward)
    explanation = explainer.attribute(data, baselines=deeplift_baseline, target=target)
    return torch.softmax(explanation, dim=1)


def lrp(model, data, target):
    explainer = LRP(model)
    return torch.softmax (explainer.attribute(data, target=target), dim=1)

def deeplift(model, data, target):
    explainer = DeepLift(model)
    explanation = explainer.attribute(data, baselines=deeplift_baseline, target=target)
    # return torch.softmax(explanation, dim=1)
    return explanation


def rexplain(model, data, target):
    res = rex(model, data[0])
    return torch.tensor([res])