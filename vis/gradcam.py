#!/usr/bin/env python3

import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def _get_layer(self, layer_name):
        """
        Return a layer (nn.Module Object) given a hierarchical layer name, separated by /.
        Args:
            layer_name (str): the name of the layer.
        """
        layer_ls = layer_name.split("/")
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        return prev_module

def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
        self, cfg, model, target_layers, data_mean, data_std, colormap="viridis"
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """
        self.cfg = cfg
        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            self.activations[layer_name] = output.clone().detach()

        target_layer = get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        self._register_single_hook(layer_name=self.target_layers)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        input_clone = inputs.clone()
        preds, logits = self.model(input_clone)

        if labels is None:
            score = torch.max(preds, dim=-1)[0]
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)

        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()

        _, _, T, H, W = inputs.size()

        gradients = self.gradients[self.target_layers]
        activations = self.activations[self.target_layers]
        B, C, Tg, _, _ = gradients.size()

        weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

        weights = weights.view(B, C, Tg, 1, 1)
        localization_map = torch.sum(
            weights * activations, dim=1, keepdim=True
        )
        localization_map = F.relu(localization_map)
        localization_map = F.interpolate(
            localization_map,
            size=(T, H, W),
            mode="trilinear",
            align_corners=False,
        )
        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                0
            ],
            torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                0
            ],
        )
        localization_map_min = torch.reshape(
            localization_map_min, shape=(B, 1, 1, 1, 1)
        )
        localization_map_max = torch.reshape(
            localization_map_max, shape=(B, 1, 1, 1, 1)
        )
        # Normalize the localization map.
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + 1e-6
        )
        localization_map = localization_map.data

        return localization_map, preds, logits

    def __call__(self, inputs, labels=None, alpha=0.5, index=None, use_labels=True):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            result_ls (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        result_ls = []
        localization_map, preds, logits = self._calculate_localization_map(
            inputs, labels=labels if use_labels else None
        )

        # Convert (B, 1, T, H, W) to (B, T, H, W)
        localization_map = localization_map.squeeze(dim=1)
        if localization_map.device != torch.device("cpu"):
            localization_map = localization_map.cpu()
        heatmap = self.colormap(localization_map.numpy())
        heatmap = heatmap[:, :, :, :, :3]
        # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
        curr_inp = inputs.permute(0, 2, 3, 4, 1)
        if curr_inp.device != torch.device("cpu"):
            curr_inp = curr_inp.cpu()
        curr_inp = revert_tensor_normalize(
            curr_inp, self.data_mean, self.data_std
        )
        heatmap = torch.from_numpy(heatmap)
        origin = curr_inp.permute(0, 1, 4, 2, 3)
        curr_inp = alpha * heatmap + (1 - alpha) * curr_inp
        # Permute inp to (B, T, C, H, W)
        curr_inp = curr_inp.permute(0, 1, 4, 2, 3)
        result_ls.append(curr_inp)
        # self.save_vis_res(index, preds, labels, origin, use_labels, origin=True, flow=True)
        # self.save_vis_res(index, preds, labels, curr_inp, use_labels)
        self.save_logits(logits, labels)

        return result_ls, preds

    def save_logits(self, logits, labels):
        if not hasattr(self, "logits"):
            self.logits = logits
            self.labels = labels
        else:
            self.logits = torch.cat((self.logits, logits), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def save_vis_res(self, vid_id, preds, labels, im_to_save, use_labels=False, origin=False, flow=False):
        # create folder
        b,t,c,h,w = im_to_save.shape
        dataset = self.cfg.TEST.DATASET
        name = self.cfg.VISUALIZATION.NAME
        for idx in range(len(labels)):
            
            if not os.path.exists('output/visualization/gradcam/{}/{}/{}/label_{:02d}'.format(
                dataset, name, self.target_layers if origin==False else "origin", labels[idx]
                )):
                os.makedirs('output/visualization/gradcam/{}/{}/{}/label_{:02d}'.format(
                    dataset, name, self.target_layers if origin==False else "origin", labels[idx]
                ))
                if flow:
                    os.makedirs('output/visualization/gradcam/{}/{}/flowx/label_{:02d}'.format(
                        dataset, name, labels[idx]
                    ))
                    os.makedirs('output/visualization/gradcam/{}/{}/flowy/label_{:02d}'.format(
                        dataset, name, labels[idx]
                    ))
                    os.makedirs('output/visualization/gradcam/{}/{}/flow/label_{:02d}'.format(
                        dataset, name, labels[idx]
                    ))

            if not use_labels:
                plt.imsave('output/visualization/gradcam/{}/{}/{}/label_{:02d}/vid_{:04d}_pred_{}.jpg'.format(
                    dataset, name, self.target_layers if origin==False else "origin", labels[idx], vid_id[idx], preds.max(1).indices[idx].detach().cpu().clone()
                ), im_to_save[idx].permute(2,0,3,1).reshape(h, -1, c).clamp(0,1).numpy())
            elif use_labels:
                plt.imsave('output/visualization/gradcam/{}/{}/{}/label_{:02d}/vid_{:04d}_{}.jpg'.format(
                    dataset, name, self.target_layers if origin==False else "origin", labels[idx], vid_id[idx], preds.max(1).indices[idx] == labels[idx]
                ), im_to_save[idx].permute(2,0,3,1).reshape(h, -1, c).clamp(0,1).numpy())

            if flow:
                optical_flow = self.extract_optical_flow(im_to_save[idx])
                optical_flow = np.transpose(optical_flow, (1,0,2,3)).reshape(h,-1,2)
                plt.imsave('output/visualization/gradcam/{}/{}/flowx/label_{:02d}/vid_{:04d}.jpg'.format(
                    dataset, name, labels[idx], vid_id[idx]
                ), optical_flow[:,:,0], cmap="jet")
                plt.imsave('output/visualization/gradcam/{}/{}/flowy/label_{:02d}/vid_{:04d}.jpg'.format(
                    dataset, name, labels[idx], vid_id[idx]
                ), optical_flow[:,:,1], cmap="jet")
                optical_flow_joint = np.sqrt(optical_flow[:,:,0] ** 2 + optical_flow[:,:,1] ** 2)
                optical_flow_joint = np.round(optical_flow_joint).astype(int)
                optical_flow_joint[optical_flow_joint>=255] = 255
                optical_flow_joint[optical_flow_joint<=0] = 0
                plt.imsave('output/visualization/gradcam/{}/{}/flow/label_{:02d}/vid_{:04d}.jpg'.format(
                    dataset, name, labels[idx], vid_id[idx]
                ), optical_flow_joint, cmap="jet")

    
    def extract_optical_flow(self, video):
        im = []
        flow = []
        b,c,h,w = video.shape
        for idx in range(b):
            im.append(
                cv2.cvtColor(video[idx].mul(255).clamp(0, 255).to(torch.uint8).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
            )
        
        for idx in range(b-1):
            flow.append(self.cal_flow(im[idx], im[idx+1]))
        flows = np.stack(flow)
        return flows

    def cal_flow(self, prev, curr, bound=15):
        if not hasattr(self, "TVL1"):
            TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        flow = (flow+bound)*(255.0/(2*bound))
        flow = np.round(flow).astype(int)
        flow[flow>=255] = 255
        flow[flow<=0] = 0
        flow = abs(flow-127)*2
        flow[flow>=255] = 255
        return flow
