from captum.attr import KernelShap, DeepLiftShap, ShapleyValueSampling, Saliency, InputXGradient, IntegratedGradients, LRP, DeepLift
import torch
from sklearn.cluster import KMeans
import numpy as np


def shapely_values(model, data):
    explainer = ShapleyValueSampling(model)
    res = explainer.attribute(
        data,
        baselines=torch.zeros_like(data)
    )
    return res

def kernel_shap(model, data):
    explainer = KernelShap(model.forward)

    explanation = explainer.attribute(
        data, 
        baselines=torch.zeros_like(data))


    return explanation

def deeplift_shap(model, data):
    explainer = DeepLiftShap(model)
    baseline = torch.zeros_like(data)

    return explainer.attribute(data, baseline)

def saliency(model, data):
    explainer = Saliency(model)

    return explainer.attribute(data)

def input_x_grad(model, data):
    explainer = InputXGradient(model.forward)
    explanation = explainer.attribute(data)
    return explanation


def integrated_grad(model, data):
    explainer = IntegratedGradients(model.forward)
    return explainer.attribute(data, baselines=torch.zeros_like(data))


def lrp(model, data):
    explainer = LRP(model)
    return explainer.attribute(data)

def deeplift(model, data):
    explainer = DeepLift(model)
    return explainer.attribute(data, baselines=torch.zeros_like(data))


