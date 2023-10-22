from logging import raiseExceptions
import torch.nn as nn

from src.models.hand_heads.hand_hmr import HandHMR
from src.models.hand_heads.mano_head import MANOHead
import core.ld_utils as ld_utils
from core.unidict import unidict
from src.nets.backbone import *

from src.models.inter_attn import inter_attn
from src.models.self_attn import SelfAttn
import torch
import time
import math

class MAHM_layer(nn.Module):
    '''
    #Mano - Attention - HMR - Mano Layer
    # input
        # mano parameters 
        # img features 
    #  intermediate processing
        # resize img features via mlp & learn separate features for each hand via single layer mlp
        # if mano vertices are used as input, compress via mlp
        # concat mano & img_features & feed to attention
        # feed attn output to HMR then to MANO
    # output
        # mano features for left and right hand
    '''
    def __init__(
        self,
        focal_length, #HMR param
        img_res, #HMR param
        img_feat_dim=2048,
        include_vert=False, #whether or not to include mesh vertices in attention layer
        downsample_image=True, #use MLP to downsample input image features
        l_r_split_from_img_feat=False, #uses MLP to learn representations for L & R hand from img_features
        vert_comp_dim=32,
        attn_feat_dim=64, #attention vector.shape[2]
        attn_out_dim=512, #size of flattened attention output
        compress_mano=None, #compress mano features prior to attention. type = int (for MLP out_dim) or None
        nheads=4, #num attention heads
        dropout=0.01,
        upsample=False #wether or not to upsample attn output features prior to HMR layer
        ):

        super().__init__()
        self.img_res = img_res
        self.nheads = nheads
        self.dropout = dropout
        self.attn_out_dim = attn_out_dim #flattened attn dim
        self.include_vert = include_vert
        self.img_feat_dim = img_feat_dim
        self.attn_feat_dim = attn_feat_dim #attention input.shape[2]
        self.downsample_image = downsample_image
        self.l_r_split_from_img_feat = l_r_split_from_img_feat
        self.compress_mano = compress_mano

        if self.include_vert:
            self.vert_comp_dim = vert_comp_dim
            assert self.vert_comp_dim % 2 ==0
            self.attn_out_dim += self.vert_comp_dim #add compressed vertex to attention output shape if vertices are used as features
            
            initial_attn_feat_dim = self.attn_feat_dim 
            if self.attn_out_dim % 8 == 0:
                self.attn_feat_dim = self.attn_out_dim // 8
            elif self.attne_out_dim % 4 == 0:
                self.attn_feat_dim = self.attn_out_dim // 4
            elif self.attne_out_dim % 2 == 0:
                self.attn_feat_dim = self.attn_out_dim // 2
            else: 
                raise ValueError('Attention dimension not divisible by 8,4, or 2 post inclusion of mesh vertices')
            
            assert self.attn_feat_dim != initial_attn_feat_dim #ensure the attention feature dimension has been resized

        else: 
            self.vert_comp_dim = 0
        
        self.attn_v_dim = self.attn_out_dim // self.attn_feat_dim #attention input.shape[1]
        
        
        if self.compress_mano != None: 
            self.mano_out_dim = self.compress_mano
        else: self.mano_out_dim = 265

        self.compressed_img_dim = self.attn_out_dim - self.mano_out_dim - self.vert_comp_dim #size of compressed image feature - mano params - compressed verticies if included #265 is the size of MANO params
                
        if self.compressed_img_dim < 0: #ensure desired attn_dim can be achieved when img_feats included
            raise ValueError('attn_feat_dim must be greater than 0 to account for mano parameters & image compression')
        
        if self.mano_out_dim < 0: #ensure desired attn_dim can be achieved when img_feats included
            raise ValueError('mano_out_dim must be greater than 0 to account for mano parameters & image compression')
        
        if self.attn_out_dim < 512: #ensure desired attn_dim can be achieved when img_feats included
            assert self.compress_mano != None #if desired attention features are less than 512 then mano compression must be enabled

        self.mlp_img_compress = nn.Linear(self.img_feat_dim, self.compressed_img_dim)
        self.mlp_left = nn.Linear(self.compressed_img_dim, self.compressed_img_dim) #adding mlp for l & r hands to learn difference from img feature
        self.mlp_right = nn.Linear(self.compressed_img_dim, self.compressed_img_dim) #adding mlp for l & r hands to learn difference from img feature

        self.mlp_compress_mano_l = nn.Linear(265, self.mano_out_dim)
        self.mlp_compress_mano_r = nn.Linear(265, self.mano_out_dim)

        self.focal_length = focal_length
        
        self.mlp_vert_compress = nn.Linear(778 * 3, self.vert_comp_dim) #adding mlp for l & r hands to learn difference from img feature

        self.position_embeddings = nn.Embedding(self.attn_v_dim, self.attn_feat_dim) 
        self.attn = inter_attn(self.attn_feat_dim, n_heads=self.nheads, dropout=self.dropout)

        self.head_r2 = HandHMR(self.attn_out_dim, is_rhand=True, n_iter=3)
        self.head_l2 = HandHMR(self.attn_out_dim, is_rhand=False, n_iter=3)
        self.mano_r2 = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )
        self.mano_l2 = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

    def forward(self, batch, mano_output_l, mano_output_r, img_feats):

        K = batch["meta.intrinsics"]

        if self.downsample_image == True: 
            img_feats = self.mlp_img_compress(img_feats)

        if self.l_r_split_from_img_feat == True:
            img_feats_left = self.mlp_left(img_feats)
            img_feats_right = self.mlp_right(img_feats)
        else:
            img_feats_left = img_feats
            img_feats_right = img_feats

        #Left hand cam, joints, beta, pose features for attn
        Lf_cam = torch.cat([mano_output_l['cam_t.wp.l'], mano_output_l['cam_t.l']], dim = 1) 
        Lf_joints = torch.cat(
            [torch.flatten(mano_output_l['joints3d.l'], start_dim = 1),
            torch.flatten(mano_output_l['joints2d.norm.l'], start_dim = 1)], dim = 1
        )
        Lf_beta = mano_output_l['beta.l']
        Lf_pose = torch.flatten(mano_output_l['pose.l'], start_dim = 1)

        Lf = torch.cat([Lf_cam, Lf_joints, Lf_beta, Lf_pose], dim = 1) #output dim = 265 + img_feats_dim
        
        if self.compress_mano != None:
            Lf = self.mlp_compress_mano_l(Lf)

        Lf = torch.cat([Lf, img_feats_left], dim = 1) #output dim = 265 + img_feats_dim


        #Right hand cam, joints, beta, pose features for attn
        Rf_cam = torch.cat([mano_output_l['cam_t.wp.l'], mano_output_l['cam_t.l']], dim = 1)
        Rf_joints = torch.cat(
            [torch.flatten(mano_output_r['joints3d.r'], start_dim = 1),
            torch.flatten(mano_output_r['joints2d.norm.r'], start_dim = 1)], dim = 1
        )
        Rf_beta = mano_output_r['beta.r']
        Rf_pose = torch.flatten(mano_output_r['pose.r'], start_dim = 1)

        Rf = torch.cat([Rf_cam, Rf_joints, Rf_beta, Rf_pose], dim = 1) #output dim = 265 + img_feats_dim
        
        if self.compress_mano != None:
            Rf = self.mlp_compress_mano_r(Rf)
        
        Rf = torch.cat([Rf, img_feats_right], dim = 1) #output dim = 265 + img_feats_dim
    
        if self.include_vert:
            Lf_vert = mano_output_l['vertices.cam.patch.l']
            Lf_vert = torch.flatten(Lf_vert, start_dim=1)
            Lf_vert = self.mlp_vert_compress(Lf_vert)
            
            Rf_vert = mano_output_r['vertices.cam.patch.r']
            Rf_vert = torch.flatten(Rf_vert, start_dim=1)
            Rf_vert = self.mlp_vert_compress(Rf_vert)

            Lf = torch.cat([Lf, Lf_vert], dim = 1)
            Rf = torch.cat([Rf, Rf_vert], dim = 1)

        Lf = torch.reshape(Lf, (-1,self.attn_v_dim, self.attn_feat_dim))
        Rf = torch.reshape(Rf, (-1,self.attn_v_dim, self.attn_feat_dim))

        BS, V, f = Lf.shape
        position_ids = torch.arange(self.attn_v_dim, dtype=torch.long, device=Lf.device)
        position_ids = position_ids.unsqueeze(0).repeat(BS, 1)
        position_embeddings = self.position_embeddings(position_ids)
        Lf = Lf + position_embeddings
        Rf = Rf + position_embeddings

        #single attention for full hand feature vectors + mobilenet features
        Lf, Rf = self.attn(Lf, Rf) #output dim = B,3,515] for Lf and Rf

        Lf = torch.flatten(Lf,start_dim=1, end_dim=-1)
        Rf = torch.flatten(Rf,start_dim=1, end_dim=-1) #flatten [B,8,64] tensor to [B,512]

        #Feed post attn features into 2nd round of HMR & Mano
        hmr_output_r2 = self.head_r2(Rf)
        hmr_output_l2 = self.head_l2(Lf)

        # weak perspective
        root_r2 = hmr_output_r2["cam_t.wp"]
        root_l2 = hmr_output_l2["cam_t.wp"]

        # decode the hand prediction
        mano_output_r2 = self.mano_r2(
            rotmat=hmr_output_r2["pose"],
            shape=hmr_output_r2["shape"],
            K=K,
            cam=root_r2,
        )

        mano_output_l2 = self.mano_l2(
            rotmat=hmr_output_l2["pose"],
            shape=hmr_output_l2["shape"],
            K=K,
            cam=root_l2,
        )

        return hmr_output_r2, hmr_output_l2, mano_output_r2, mano_output_l2

            


class IHModel(nn.Module):
    def __init__(
        self,
        focal_length,
        img_res,
    ):
        super().__init__()
        self.backbone = eval("resnet50")(pretrained=True)
        #self.mobile_net = mobilenet_v2(pretrained = True, progress = False) #used for attn
        feat_dim = 2048
        self.compress_resnet = nn.Linear(in_features = 2048, out_features = 247) #out_feat chosen to yield attn input dim of 512

        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length
        self.pool = nn.AdaptiveAvgPool2d(1)

        #==================================================================#
        #         Attention Cascade - currently 3 layer                    #
        #==================================================================#
        
        attn_out_dim = [1024, 512, 256, 128, 256, 512, 1024]
        attn_feat_dim = [128, 64, 32, 16, 32, 64, 128]
        mano_compression = [None, None, 128, 64, 128, None, None]

        self.layers = nn.ModuleList()
        for i in range(len(attn_out_dim)):
            self.layers.append(MAHM_layer(focal_length=focal_length, #HMR param
            img_res=img_res, #HMR param
            img_feat_dim=2048,
            include_vert=False, #whether or not to include mesh vertices in attention layer
            downsample_image=True, #use MLP to downsample input image features
            l_r_split_from_img_feat=True, #uses MLP to learn representations for L & R hand from img_features
            vert_comp_dim=32,
            attn_feat_dim=attn_feat_dim[i], #attention vector.shape[2]
            attn_out_dim=attn_out_dim[i], #size of flattened attention output
            compress_mano=mano_compression[i], #compress mano features prior to attention. type = int (for MLP out_dim) or None
            nheads=4, #num attention heads
            dropout=0.01,
            upsample=False #wether or not to upsample attn output features prior to HMR layer)
            ))

    def forward(self, batch):
        images = batch["inputs.images"]
        K = batch["meta.intrinsics"]
        features = self.backbone(images)
        features = self.pool(features).view(-1, features.shape[1])

        # predict cameras and hand poses with HMR refinement
        hmr_output_r = self.head_r(features)
        hmr_output_l = self.head_l(features)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]

        # decode the hand prediction
        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        #==================================================================#
        #         Attention Cascade - currently double layer               #
        #==================================================================#
        for i in range(len(self.layers)):
            hmr_output_r2, hmr_output_l2, mano_output_r2, mano_output_l2 = self.layers[i](batch, mano_output_l, mano_output_r, features)

        # fwd mesh when in val or vis
        root_r_init = hmr_output_r2["cam_t.wp.init"]
        root_l_init = hmr_output_l2["cam_t.wp.init"]
        mano_output_r2.register("cam_t.wp.init.r", root_r_init)
        mano_output_l2.register("cam_t.wp.init.l", root_l_init)

        mano_output_r2 = ld_utils.prefix_dict(mano_output_r2, "mano.")
        mano_output_l2 = ld_utils.prefix_dict(mano_output_l2, "mano.")
        output = unidict()
        output.merge(mano_output_r2)
        output.merge(mano_output_l2)
        return output
