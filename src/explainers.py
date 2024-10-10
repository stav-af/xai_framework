from captum.attr import KernelShap
import torch

def shap_explainer(model, data):
    explainer = KernelShap(model.forward)
    explanation = explainer.attribute(torch.tensor(data, dtype=torch.float32, requires_grad=True), perturbations_per_eval=500).tolist()

    return explanation


