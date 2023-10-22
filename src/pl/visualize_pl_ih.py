import trimesh
import numpy as np
import matplotlib.pyplot as plt

from src.utils.image_utils import denormalize_images
from src.utils.rend import color2material
import core.vis_utils as vis_utils
import core.torch_cam_utils as tcu


mesh_color_dict = {
    "right": [200, 200, 250],
    "left": [100, 100, 250],
    "top": [144, 250, 100],
    "bottom": [129, 159, 214],
}


def im_list_to_plt(image_list, figsize, title_list=None):
    fig, axes = plt.subplots(nrows=1, ncols=len(image_list), figsize=figsize)
    for idx, (ax, im) in enumerate(zip(axes, image_list)):
        ax.imshow(im)
        ax.set_title(title_list[idx])
    fig.tight_layout()
    im = vis_utils.fig2img(fig)
    plt.close()
    return im


def visualize_pred_gt(
    images_i,
    joints2d_r_i_gt,
    joints2d_l_i_gt,
    joints2d_proj_r_i_gt,
    joints2d_proj_l_i_gt,
    joints2d_r_i_pred,
    joints2d_l_i_pred,
    joints2d_proj_r_i_pred,
    joints2d_proj_l_i_pred,
):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax = ax.reshape(-1)

    ax[0].scatter(joints2d_r_i_gt[:, 0], joints2d_r_i_gt[:, 1], color="r", marker="x")
    ax[0].scatter(joints2d_l_i_gt[:, 0], joints2d_l_i_gt[:, 1], color="b", marker="x")
    ax[0].imshow(images_i)
    ax[0].set_title("GT 2D keypoints")

    ax[1].scatter(
        joints2d_r_i_pred[:, 0], joints2d_r_i_pred[:, 1], color="r", marker="x"
    )
    ax[1].scatter(
        joints2d_l_i_pred[:, 0], joints2d_l_i_pred[:, 1], color="b", marker="x"
    )
    ax[1].imshow(images_i)
    ax[1].set_title("Pred 2D keypoints")
    ax[1].imshow(images_i)
    ax[1].set_title("Pred 2D keypoints")

    ax[2].scatter(
        joints2d_proj_r_i_gt[:, 0], joints2d_proj_r_i_gt[:, 1], color="r", marker="x"
    )
    ax[2].scatter(
        joints2d_proj_l_i_gt[:, 0], joints2d_proj_l_i_gt[:, 1], color="b", marker="x"
    )
    ax[2].imshow(images_i)
    ax[2].set_title("GT 3D keypoints reprojection from cam")

    ax[3].scatter(
        joints2d_proj_r_i_pred[:, 0],
        joints2d_proj_r_i_pred[:, 1],
        color="r",
        marker="x",
    )
    ax[3].scatter(
        joints2d_proj_l_i_pred[:, 0],
        joints2d_proj_l_i_pred[:, 1],
        color="b",
        marker="x",
    )
    ax[3].imshow(images_i)
    ax[3].set_title("Pred 3D keypoints reprojection from cam")

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout()
    plt.close()
    im = vis_utils.fig2img(fig)
    return im


def compare_gts_preds(vis_dict):
    images = (vis_dict["vis.images"].permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)

    K = vis_dict["meta_info.intrinsics"]

    flag = "targets"
    # the original 2d keypoints projected using perspective camera parameters
    # this usually has better alignment than the model prediction (which is using a weak perspective camera)
    gt_joints2d_r = vis_dict[f"{flag}.mano.joints2d.r"].numpy()
    gt_joints2d_l = vis_dict[f"{flag}.mano.joints2d.l"].numpy()

    # project the GT 3d keypoints to 2d
    gt_joints3d_r = vis_dict[f"{flag}.mano.joints3d.cam.patch.r"]
    gt_joints2d_proj_r = tcu.project2d_batch(K, gt_joints3d_r).numpy()
    gt_joints3d_l = vis_dict[f"{flag}.mano.joints3d.cam.patch.l"]
    gt_joints2d_proj_l = tcu.project2d_batch(K, gt_joints3d_l).numpy()

    flag = "pred"
    pred_joints2d_r = vis_dict[f"{flag}.mano.joints2d.r"].numpy()
    pred_joints2d_l = vis_dict[f"{flag}.mano.joints2d.l"].numpy()

    # project the predicted 3d keypoints to 2d
    pred_joints3d_r = vis_dict[f"{flag}.mano.joints3d.cam.patch.r"]
    pred_joints2d_proj_r = tcu.project2d_batch(K, pred_joints3d_r).numpy()
    pred_joints3d_l = vis_dict[f"{flag}.mano.joints3d.cam.patch.l"]
    pred_joints2d_proj_l = tcu.project2d_batch(K, pred_joints3d_l).numpy()

    im_list = []
    for idx in range(min(images.shape[0], 10)):
        image_id = vis_dict["vis.image_ids"][idx]
        im = visualize_pred_gt(
            images[idx],
            gt_joints2d_r[idx],
            gt_joints2d_l[idx],
            gt_joints2d_proj_r[idx],
            gt_joints2d_proj_l[idx],
            pred_joints2d_r[idx],
            pred_joints2d_l[idx],
            pred_joints2d_proj_r[idx],
            pred_joints2d_proj_l[idx],
        )
        im_list.append({"fig_name": f"{image_id}__keypoints_{flag}", "im": im})
    return im_list


def render_result(
    renderer,
    vertices_r,
    vertices_l,
    mano_faces_r,
    mano_faces_l,
    r_valid,
    l_valid,
    K,
    img,
):
    img = img.permute(1, 2, 0).cpu().numpy()
    mesh_r = trimesh.Trimesh(vertices_r, mano_faces_r)
    mesh_l = trimesh.Trimesh(vertices_l, mano_faces_l)

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid:
        meshes.append(mesh_l)
        mesh_names.append("left")

    render_img_img = render_meshes(
        renderer, meshes, mesh_names, K, img, sideview_angle=None
    )
    render_img_angles = []
    for angle in list(np.linspace(45, 300, 3)):
        render_img_angle = render_meshes(
            renderer, meshes, mesh_names, K, img=None, sideview_angle=angle
        )
        render_img_angles.append(render_img_angle)
    render_img_angles = [render_img_img] + render_img_angles
    render_img = np.concatenate(render_img_angles, axis=0)
    return render_img


def render_meshes(renderer, meshes, mesh_names, K, img, sideview_angle):
    materials = [color2material(mesh_color_dict[name]) for name in mesh_names]
    rend_img = (
        renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=sideview_angle,
            K=K,
        )
        * 255
    )
    rend_img = rend_img.astype(np.uint8)
    return rend_img


def visualize_all(vis_dict, max_examples, renderer, postfix, no_tqdm):
    K = vis_dict["meta_info.intrinsics"]
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    vis_dict.remove("inputs.img")
    vis_dict.register("vis.images", images)
    vis_dict.register("vis.image_ids", image_ids)

    # unpack MANO
    pred_vertices_r_cam = vis_dict["pred.mano.vertices.cam.patch.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.vertices.cam.patch.l"]
    gt_vertices_r_cam = vis_dict["targets.mano.vertices.cam.patch.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.vertices.cam.patch.l"]

    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]

    # valid flag
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()

    im_list = []
    # rendering
    for idx in range(min(len(image_ids), 16)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        if not (l_valid or r_valid):
            continue
        K_i = K[idx]
        image_id = image_ids[idx]

        # meshes
        image_list = []
        image_list.append(images[idx].permute(1, 2, 0).cpu().numpy())
        image_gt = render_result(
            renderer,
            gt_vertices_r_cam[idx],
            gt_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            r_valid,
            l_valid,
            K_i,
            images[idx],
        )
        image_list.append(image_gt)

        # prediction
        image_pred = render_result(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            r_valid,
            l_valid,
            K_i,
            images[idx],
        )
        image_list.append(image_pred)

        image_pred = im_list_to_plt(
            image_list,
            figsize=(15, 8),
            title_list=["input image", "GT", "pred"],
        )
        im_list.append({"fig_name": f"{image_id}__render", "im": image_pred})

    # visualize GT and predicted keypoints
    im_list += compare_gts_preds(vis_dict)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)
    return im_list
