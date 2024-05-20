#class Octopus():
    #def __init__(self):

from Running import inference

def octopus(input,target_models, tokenizer, device):
    head_classifier = ...
    output = 0
    for target, weight in head_classifier:
        model_output = inference(model=target_models[target], input = input, tokenizer = tokenizer, device= device)
        print("using the model specified on",target," outputs", model_output)
        output += model_output*weight
    return output