from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
import os


def visualize_record(record, meta_data, suptitle=""):
    img = (
        record["image"].numpy().transpose(1, 2, 0)
        if "image" in record
        else cv2.imread(record["file_name"])
    )
    visualizer = Visualizer(img, metadata=meta_data, scale=1)
    image_with_gt = visualizer.draw_dataset_dict(record).get_image()
    plot_side_by_side(
        imgs=[img[:, :, ::-1], image_with_gt[:, :, ::-1]],
        titles=["Raw Image", "Image with Ground Truth"],
        suptitle=suptitle,
    )


def plot_one_image(img, title, show_plot=True, save_path=None):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.tight_layout()
    plt.xticks([]), plt.yticks([])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()


def plot_side_by_side(imgs, titles, show_plot=True, save_path=None, suptitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(suptitle, fontsize=18)
    for ax in axes:
        ax.set_xticklabels([]), ax.set_yticklabels([])
    axes[0].imshow(imgs[0])
    axes[0].set_title(titles[0], fontdict={"size": 14})
    axes[1].imshow(imgs[1])
    axes[1].set_title(titles[1], fontdict={"size": 14})
    fig.tight_layout()
    # fig.subplots_adjust(top=0.92)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
