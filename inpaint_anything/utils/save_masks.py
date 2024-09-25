from pathlib import Path
from matplotlib import pyplot as plt
from inpaint_anything.utils.utils import save_array_to_img, show_mask, show_points


def save_masks(point_coords, out_dir, img, mask, point_labels, img_inpainted, endpoint_name):
    mask_p = out_dir / f"mask.png"
    img_points_p = out_dir / f"with_points.png"
    img_mask_p = out_dir / f"with_{Path(mask_p).name}"

    # Save the mask
    save_array_to_img(mask, mask_p)

    # Save the pointed and masked image
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), [point_coords], point_labels,
                size=(width*0.04)**2)
    plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
    show_mask(plt.gca(), mask, random_color=False)
    plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    img_inpainted_p = out_dir / f"{endpoint_name}_{Path(mask_p).name}"
    save_array_to_img(img_inpainted, img_inpainted_p)