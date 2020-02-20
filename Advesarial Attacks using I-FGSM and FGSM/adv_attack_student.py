"""
Adversary Attack example code
"""
# from torchvision import models as tvm
import torch
from torch.nn import functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from Model import Net


class AdversialAttacker(object):

    def __init__(self, method='FGSM'):
        assert method in ['FGSM', 'I-FGSM']
        self.method = method
        self.criterion = torch.nn.CrossEntropyLoss()
        print("created adversial attacker in method '%s'" % (method))

    def get_pred_label(self, mdl, inp, ret_out_scores=False, ret_out_pred=True):
        # use current model to get predicted label
        train = mdl.training
        mdl.eval()
        with torch.no_grad():
            out = F.softmax(mdl(inp), dim=1)
        out_score, out_pred = out.max(dim=1)
        if ret_out_scores and not ret_out_pred:
            return out
        if ret_out_pred and not ret_out_scores:
            return out_pred
        mdl.train(train)
        return out_pred, out

    def perturb_untargeted(self, mdl, inp, targ_label=None, eps=0.3):
        # perform attacking perturbation in the untargeted setting
        # note: feel free the change the function arguments for your implementation
        mdl.train()  # switch model to train mode
        out_pred, out_score=self.get_pred_label(mdl,inp,True,True)
        x_adv = Variable(inp.data, requires_grad=True)
        if self.method == 'FGSM':
            h_adv = mdl(x_adv)
            if targ_label:
                cost = self.criterion(h_adv, targ_label)
            else:
                cost = -self.criterion(h_adv, out_pred)

            mdl.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - eps * x_adv.grad

            h = mdl(inp)
            h_adv = mdl(x_adv)

            return x_adv, h_adv, h

        elif self.method == 'I-FGSM':
            iteration=5
            alpha=1
            # TODO
            # you may add arguments like iter, eps_iter, ...
            x_adv = Variable(inp.data, requires_grad=True)
            for i in range(iteration):
                h_adv = mdl(x_adv)

                cost = -self.criterion(h_adv, targ_label)

                self.net.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                cost.backward()

                x_adv.grad.sign_()
                x_adv = x_adv - alpha * x_adv.grad
                x_adv = np.where(x_adv > inp + eps, inp + eps, x_adv)
                x_adv = np.where(x_adv < inp - eps, inp - eps, x_adv)
                x_adv = Variable(x_adv.data, requires_grad=True)

            h_adv = mdl(x_adv)

        mdl.eval()  # switch model back
        # return the attacked image tensor
        return h_adv

    def perturb_targeted(self, mdl, inp, targ_label, eps=0.3):
        # perform attacking perturbation in the targeted setting
        # note: feel free the change the function arguments for your implementation
        mdl.train()  # switch model to train mode

        if self.method == 'FGSM':
            pass
            # TODO

        elif self.method == 'I-FGSM':
            pass
            # TODO
            # you may add arguments like iter, eps_iter, ...

        mdl.eval()  # switch model back
        # return the attacked image tensor
        return inp_adv


class Clamp:
    def __call__(self, inp):
        return torch.clamp(inp, 0., 1.)


def generate_experiment(method='FGSM'):

    # define your model and load pretrained weights
    # TODO
    # model = ...
    model = Net()
    model=model.load_state_dict(torch.load("/content/drive/My Drive/Colab Notebooks/model64"))

    # cinic class names
    import yaml
    with open('./cinic_classnames.yml', 'r') as fp:
        classnames = yaml.safe_load(fp)

    # load image
    # TODO:
    # img_path = Path(....)
    img_path=Path("/content/test/")
    input_img = Image.open(img_path/"airplane/cifar10-test-10.png")

    # define normalizer and un-normalizer for images
    # cinic
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    tf_img = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
    )
    un_norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            ),
            Clamp(),
            transforms.ToPILImage()
        ]
    )

    # To be used for iterative method
    # to ensure staying within Linf limits
    clip_min = min([-m/s for m, s in zip(mean, std)])
    clip_max = max([(1-m)/s for m, s in zip(mean, std)])

    input_tensor = tf_img(input_img)
    attacker = AdversialAttacker(method=method)

    return {
        'img': input_img,
        'inp': input_tensor.unsqueeze(0),
        'attacker': attacker,
        'mdl': model,
        'clip_min': clip_min,
        'clip_max': clip_max,
        'un_norm': un_norm,
        'classnames': classnames
    }
