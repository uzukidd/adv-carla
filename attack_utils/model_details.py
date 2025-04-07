class model_details:
    pointpillar = """
    PointPillar(
        (vfe): PillarVFE(
            (pfn_layers): ModuleList(
            (0): PFNLayer(
                (linear): Linear(in_features=10, out_features=64, bias=False)
                (norm): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
            )
        )
        (backbone_3d): None
        (map_to_bev_module): PointPillarScatter()
        (pfe): None
        (backbone_2d): BaseBEVBackbone(
            (blocks): ModuleList(
            (0): Sequential(
                (0): ZeroPad2d((1, 1, 1, 1))
                (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (3): ReLU()
                (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (6): ReLU()
                (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (9): ReLU()
                (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (12): ReLU()
            )
            (1): Sequential(
                (0): ZeroPad2d((1, 1, 1, 1))
                (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (3): ReLU()
                (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (6): ReLU()
                (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (9): ReLU()
                (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (12): ReLU()
                (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (15): ReLU()
                (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (18): ReLU()
            )
            (2): Sequential(
                (0): ZeroPad2d((1, 1, 1, 1))
                (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
                (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (3): ReLU()
                (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (6): ReLU()
                (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (9): ReLU()
                (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (12): ReLU()
                (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (15): ReLU()
                (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (18): ReLU()
            )
            )
            (deblocks): ModuleList(
            (0): Sequential(
                (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU()
            )
            (1): Sequential(
                (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
                (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU()
            )
            (2): Sequential(
                (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
                (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU()
            )
            )
        )
        (dense_head): AnchorHeadSingle(
            (cls_loss_func): SigmoidFocalClassificationLoss()
            (reg_loss_func): WeightedSmoothL1Loss()
            (dir_loss_func): WeightedCrossEntropyLoss()
            (conv_cls): Conv2d(384, 18, kernel_size=(1, 1), stride=(1, 1))
            (conv_box): Conv2d(384, 42, kernel_size=(1, 1), stride=(1, 1))
            (conv_dir_cls): Conv2d(384, 12, kernel_size=(1, 1), stride=(1, 1))
        )
        (point_head): None
        (roi_head): None
        )
    """

    pointrcnn = """
    PointRCNN(
    (vfe): None
    (backbone_3d): PointNet2MSG(
        (SA_modules): ModuleList(
        (0): PointnetSAModuleMSG(
            (groupers): ModuleList(
            (0-1): 2 x QueryAndGroup()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            (1): Sequential(
                (0): Conv2d(4, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        (1): PointnetSAModuleMSG(
            (groupers): ModuleList(
            (0-1): 2 x QueryAndGroup()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            (1): Sequential(
                (0): Conv2d(99, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(64, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        (2): PointnetSAModuleMSG(
            (groupers): ModuleList(
            (0-1): 2 x QueryAndGroup()
            )
            (mlps): ModuleList(
            (0-1): 2 x Sequential(
                (0): Conv2d(259, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(128, 196, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(196, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        (3): PointnetSAModuleMSG(
            (groupers): ModuleList(
            (0-1): 2 x QueryAndGroup()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            (1): Sequential(
                (0): Conv2d(515, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(384, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        )
        (FP_modules): ModuleList(
        (0): PointnetFPModule(
            (mlp): Sequential(
            (0): Conv2d(257, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            )
        )
        (1): PointnetFPModule(
            (mlp): Sequential(
            (0): Conv2d(608, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            )
        )
        (2): PointnetFPModule(
            (mlp): Sequential(
            (0): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            )
        )
        (3): PointnetFPModule(
            (mlp): Sequential(
            (0): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            )
        )
        )
    )
    (map_to_bev_module): None
    (pfe): None
    (backbone_2d): None
    (dense_head): None
    (point_head): PointHeadBox(
        (cls_loss_func): SigmoidFocalClassificationLoss()
        (reg_loss_func): WeightedSmoothL1Loss()
        (cls_layers): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=256, out_features=256, bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=256, out_features=3, bias=True)
        )
        (box_layers): Sequential(
        (0): Linear(in_features=128, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Linear(in_features=256, out_features=256, bias=False)
        (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Linear(in_features=256, out_features=8, bias=True)
        )
    )
    (roi_head): PointRCNNHead(
        (proposal_target_layer): ProposalTargetLayer()
        (reg_loss_func): WeightedSmoothL1Loss()
        (SA_modules): ModuleList(
        (0): PointnetSAModule(
            (groupers): ModuleList(
            (0): QueryAndGroup()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        (1): PointnetSAModule(
            (groupers): ModuleList(
            (0): QueryAndGroup()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        (2): PointnetSAModule(
            (groupers): ModuleList(
            (0): GroupAll()
            )
            (mlps): ModuleList(
            (0): Sequential(
                (0): Conv2d(259, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (5): ReLU()
                (6): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (8): ReLU()
            )
            )
        )
        )
        (xyz_up_layer): Sequential(
        (0): Conv2d(5, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (3): ReLU()
        )
        (merge_down_layer): Sequential(
        (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU()
        )
        (cls_layers): Sequential(
        (0): Conv1d(512, 256, kernel_size=(1,), stride=(1,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
        )
        (reg_layers): Sequential(
        (0): Conv1d(512, 256, kernel_size=(1,), stride=(1,), bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.0, inplace=False)
        (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv1d(256, 7, kernel_size=(1,), stride=(1,))
        )
        (roipoint_pool3d_layer): RoIPointPool3d()
    )
    )
    
    """