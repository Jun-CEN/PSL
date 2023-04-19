#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

from models.module_zoo.branches.r2plus1d_branch import R2Plus1DBranch
from models.module_zoo.branches.r2d3d_branch import R2D3DBranch
from models.module_zoo.branches.csn_branch import CSNBranch
from models.module_zoo.branches.msm_branch import MSMBranch
from models.module_zoo.branches.msm_branch_v2 import MultiscaleAggregatorSpatialTemporal
from models.module_zoo.branches.slowfast_branch import SlowfastBranch
from models.module_zoo.branches.x3d_branch import X3DBranch

from models.module_zoo.branches.s3dg_branch import STConv3d

from models.module_zoo.branches.non_local import NonLocal

from models.module_zoo.branches.condconv import CondConvBaseline
from models.module_zoo.branches.temporal_adaptive_spatialconv import TemporalAdaptiveSpatialConvBlock
import models.module_zoo.branches.patched_tada
import models.module_zoo.branches.tada_av
import models.module_zoo.branches.tada_av_v2