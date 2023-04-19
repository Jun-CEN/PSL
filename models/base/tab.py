
import torch
import torch.nn as nn
from models.base.models import MODEL_REGISTRY
from models.utils.init_helper import _init_convnet_weights, _init_transformer_weights

class NonLocalBlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_dropout=0.,
        ff_dropout=0.,
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_q           = nn.Linear(dim, dim)
        self.to_kv          = nn.Linear(dim, dim * 2)
        self.attn_dropout   = nn.Dropout(attn_dropout)

        self.norm           = nn.LayerNorm(dim, eps=1e-6)
        self.relu           = nn.ReLU(inplace=True)

        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)



    def forward(self, x1, x2):
        B, N1, C = x1.shape
        B, N2, C = x2.shape
        q = self.to_q(x1).reshape(B, N1, C)
        kv = self.to_kv(x2).reshape(B, N2, 2, C).permute(2, 0, 1, 3)
        k = kv[0]
        v = kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)

        out = self.norm(out)
        out = self.relu(out)

        out = self.proj(out)
        out = self.ff_dropout(out)
        return x1 + out

@MODEL_REGISTRY.register()
class TemporalAggregateNet(nn.Module):

    def __init__(self, cfg):
        super(TemporalAggregateNet, self).__init__()
        self.cfg = cfg

        if cfg.VIDEO.INPUT_FEATURE == "vivit":
            feature_dim = 768 * cfg.VIDEO.NUM_INPUT_VIEWS
        elif cfg.VIDEO.INPUT_FEATURE == "slowfast":
            feature_dim = 2304 * cfg.VIDEO.NUM_INPUT_VIEWS
        else:
            feature_dim = 2048 * cfg.VIDEO.NUM_INPUT_VIEWS

            
        self.tab_1 = TemporalAggregateBlock(cfg)
        self.tab_2 = TemporalAggregateBlock(cfg)

        self.sk_maxpools = [nn.AdaptiveMaxPool1d((int(i))) for i in cfg.VIDEO.SK_SCALES]
        rk_start = (cfg.DATA.TEMPORAL_SCALE-cfg.VIDEO.RK_LENGTH_TOTAL) // 2
        rk_end = rk_start + cfg.VIDEO.RK_LENGTH_TOTAL
        self.rk_indexes = torch.linspace(rk_start, rk_end-1, cfg.VIDEO.RK_LENGTH_TOTAL).long()
        self.rk_maxpool = nn.AdaptiveMaxPool1d(cfg.VIDEO.RK_SCALE)

        self.linear_verb_1 = nn.Linear(feature_dim, cfg.VIDEO.HEAD.NUM_CLASSES[0])
        self.linear_verb_2 = nn.Linear(feature_dim, cfg.VIDEO.HEAD.NUM_CLASSES[0])

        self.linear_noun_1 = nn.Linear(feature_dim, cfg.VIDEO.HEAD.NUM_CLASSES[1])
        self.linear_noun_2 = nn.Linear(feature_dim, cfg.VIDEO.HEAD.NUM_CLASSES[1])

        self.apply(_init_transformer_weights)
        
    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]
        spanning_past_list = self.divide_sk(x)
        recent_past_list = self.divide_rk(x)

        out = []
        for recent_past in recent_past_list:
            spanning_past, recent_past = self.tab_1(spanning_past_list, recent_past)
            out.append(
                torch.cat((spanning_past, recent_past), dim=1).mean(1, keepdim=True)
            )

        verb_pred = self.linear_verb_1(out[0]).softmax(-1) + self.linear_verb_2(out[1]).softmax(-1)
        noun_pred = self.linear_noun_1(out[0]).softmax(-1) + self.linear_noun_2(out[1]).softmax(-1)
        if not self.training:
            verb_pred = verb_pred.softmax(-1)
            noun_pred = noun_pred.softmax(-1)
        verb_pred = verb_pred.view(verb_pred.shape[0], -1)
        noun_pred = noun_pred.view(noun_pred.shape[0], -1)

        return {"verb_class": verb_pred, "noun_class": noun_pred}, out

    def divide_sk(self, x):
        return [self.sk_maxpools[i](x).permute(0, 2, 1) for i in range(len(self.sk_maxpools))]

    def divide_rk(self, x):
        return [
            self.rk_maxpool(x[:, :, self.rk_indexes[:self.cfg.VIDEO.RK_LENGTH_EACH]]).permute(0, 2, 1),
            self.rk_maxpool(x[:, :, self.rk_indexes[self.cfg.VIDEO.RK_LENGTH_EACH:]]).permute(0, 2, 1)
        ]
            

class TemporalAggregateBlock(nn.Module):
    def __init__(self, cfg):
        super(TemporalAggregateBlock, self).__init__()
        self.cfg = cfg

        if cfg.VIDEO.INPUT_FEATURE == "vivit":
            feature_dim = 768 * cfg.VIDEO.NUM_INPUT_VIEWS
        elif cfg.VIDEO.INPUT_FEATURE == "slowfast":
            feature_dim = 2304 * cfg.VIDEO.NUM_INPUT_VIEWS
        else:
            feature_dim = 2048 * cfg.VIDEO.NUM_INPUT_VIEWS

        self.cp1 = CouplingBlock(cfg)
        self.cp2 = CouplingBlock(cfg)
        self.cp3 = CouplingBlock(cfg)

        self.linear = nn.Linear(feature_dim, feature_dim)
        

    def forward(self, x_spanning, x_recent):
        x_recent_out_all = []
        x_spanning_out_all = []
        for i, x_sp in enumerate(x_spanning):
            x_spanning_out, x_recent_out = getattr(self, f"cp{i+1}")(x_sp, x_recent)
            x_recent_out_all.append(x_recent_out)
            x_spanning_out_all.append(x_spanning_out)
        x_recent_out_all = torch.cat(x_recent_out_all, dim=1).mean(1, keepdim=True)
        x_recent_out_all = self.linear(x_recent_out_all)
        x_spanning_out_all = torch.cat(x_spanning_out_all, dim=1)
        x_spanning_out_all = x_spanning_out_all.max(1, keepdim=True)[0]

        return x_spanning_out_all, x_recent_out_all

class CouplingBlock(nn.Module):

    def __init__(self, cfg):
        super(CouplingBlock, self).__init__()
        self.cfg = cfg
        if cfg.VIDEO.INPUT_FEATURE == "vivit":
            feature_dim = 768 * cfg.VIDEO.NUM_INPUT_VIEWS
        elif cfg.VIDEO.INPUT_FEATURE == "slowfast":
            feature_dim = 2304 * cfg.VIDEO.NUM_INPUT_VIEWS
        else:
            feature_dim = 2048 * cfg.VIDEO.NUM_INPUT_VIEWS
        self.nlb_1 = NonLocalBlock(
            feature_dim, 
            cfg.VIDEO.NON_LOCAL.DROPOUT,
            cfg.VIDEO.NON_LOCAL.DROPOUT,
        )

        self.nlb_2 = NonLocalBlock(
            feature_dim, 
            cfg.VIDEO.NON_LOCAL.DROPOUT,
            cfg.VIDEO.NON_LOCAL.DROPOUT,
        )

        self.linear_recent = nn.Linear(feature_dim, feature_dim)
        self.linear_spanning = nn.Linear(feature_dim, feature_dim)

    def forward(self, x_spanning, x_recent):
        x_spanning = self.nlb_1(x_spanning, x_spanning)
        x_out = self.nlb_2(x_recent, x_spanning)

        x_spanning = torch.cat((x_spanning, x_out), dim=1)
        x_spanning = self.linear_spanning(x_spanning.mean(1, keepdim=True))

        x_recent = torch.cat((x_recent, x_out), dim=1)
        x_recent = self.linear_recent(x_recent.mean(1,keepdim=True))
        return x_spanning, x_recent


@MODEL_REGISTRY.register()
class DebugNet(nn.Module):

    def __init__(self, cfg):
        super(DebugNet, self).__init__()
        self.cfg = cfg

        if cfg.VIDEO.INPUT_FEATURE == "vivit":
            feature_dim = 768 * cfg.VIDEO.NUM_INPUT_VIEWS
        elif cfg.VIDEO.INPUT_FEATURE == "slowfast":
            feature_dim = 2304 * cfg.VIDEO.NUM_INPUT_VIEWS
        else:
            feature_dim = 2048 * cfg.VIDEO.NUM_INPUT_VIEWS

        num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        temporal_len = cfg.DATA.TEMPORAL_SCALE
        mid_dim = 1024

        self.head_conv_verb = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, num_classes[0], kernel_size=1, padding=0),
                )
        self.head_conv_noun = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, num_classes[1], kernel_size=1, padding=0),
                )
        
    def forward(self, x):
        if isinstance(x, dict):
            x = x["video"]

        verb_pred = self.head_conv_verb(x)
        noun_pred = self.head_conv_noun(x)
        if not self.training:
            verb_pred = verb_pred.softmax(-2)
            noun_pred = noun_pred.softmax(-2)
        verb_pred = verb_pred.view(verb_pred.shape[0], -1)
        noun_pred = noun_pred.view(noun_pred.shape[0], -1)

        return {"verb_class": verb_pred, "noun_class": noun_pred}, x.mean(dim=2)

@MODEL_REGISTRY.register()
class DebugNetObj(nn.Module):

    def __init__(self, cfg):
        super(DebugNetObj, self).__init__()
        self.cfg = cfg

        if cfg.VIDEO.INPUT_FEATURE == "vivit":
            feature_dim = 768 * cfg.VIDEO.NUM_INPUT_VIEWS
        elif cfg.VIDEO.INPUT_FEATURE == "slowfast":
            feature_dim = 2304 * cfg.VIDEO.NUM_INPUT_VIEWS
        else:
            feature_dim = 2048 * cfg.VIDEO.NUM_INPUT_VIEWS

        num_classes = cfg.VIDEO.HEAD.NUM_CLASSES
        temporal_len = cfg.DATA.TEMPORAL_SCALE
        mid_dim = 1024

        self.obj_preprocess = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1, padding=0, groups=4, stride=1),
            nn.BatchNorm1d(512, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, mid_dim, kernel_size=1, padding=0, groups=4, stride=1)
        )


        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x":
            self.head_conv_verb = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                )
            self.head_conv_noun = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                )

            self.post_process_verb = nn.Sequential(
                nn.BatchNorm1d(mid_dim*2),
                nn.ReLU(inplace=True),
            )
            self.post_process_noun = nn.Sequential(
                nn.BatchNorm1d(mid_dim*2),
                nn.ReLU(inplace=True),
            )
            
            self.pred_conv_verb = nn.Conv1d(mid_dim*2, num_classes[0], kernel_size=1, padding=0)
            self.pred_conv_noun = nn.Conv1d(mid_dim*2, num_classes[1], kernel_size=1, padding=0)
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun":
            self.head_conv_verb = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                )
            self.head_conv_noun = nn.Sequential(
                nn.Conv1d(feature_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=4, stride=2),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=temporal_len//4, padding=0, groups=4),
                )

            self.post_process_noun = nn.Sequential(
                nn.BatchNorm1d(mid_dim*2),
                nn.ReLU(inplace=True),
            )
            
            self.pred_conv_verb = nn.Conv1d(mid_dim, num_classes[0], kernel_size=1, padding=0)
            self.pred_conv_noun = nn.Conv1d(mid_dim*2, num_classes[1], kernel_size=1, padding=0)
        
    def forward(self, x):
        if isinstance(x, dict):
            x_vid = x["video"]
            x_obj = x["obj_feat"].permute(0,2,1)

        if self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_x":
            obj_feat = self.obj_preprocess(x_obj).mean(-1, keepdim=True)
            
            verb_feat = self.head_conv_verb(x_vid)
            noun_feat = self.head_conv_noun(x_vid)

            verb_feat = self.post_process_verb(torch.cat((verb_feat, obj_feat), dim=1))
            noun_feat = self.post_process_noun(torch.cat((noun_feat, obj_feat), dim=1))

            verb_pred = self.pred_conv_verb(verb_feat)
            noun_pred = self.pred_conv_noun(noun_feat)
        elif self.cfg.VIDEO.HEAD.OBJ_FEAT == "cat_noun":
            obj_feat = self.obj_preprocess(x_obj).mean(-1, keepdim=True)
            
            verb_feat = self.head_conv_verb(x_vid)
            noun_feat = self.head_conv_noun(x_vid)

            noun_feat = self.post_process_noun(torch.cat((noun_feat, obj_feat), dim=1))

            verb_pred = self.pred_conv_verb(verb_feat)
            noun_pred = self.pred_conv_noun(noun_feat)


        if not self.training:
            verb_pred = verb_pred.softmax(-2)
            noun_pred = noun_pred.softmax(-2)
        verb_pred = verb_pred.view(verb_pred.shape[0], -1)
        noun_pred = noun_pred.view(noun_pred.shape[0], -1)

        return {"verb_class": verb_pred, "noun_class": noun_pred}, x["video"].mean(dim=2)