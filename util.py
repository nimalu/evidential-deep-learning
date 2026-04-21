import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import matplotlib.tri as tri
from matplotlib.widgets import Slider
from scipy.special import gamma, psi
import torch
from typing import List, Optional
import matplotlib.gridspec as gridspec


CIFAR_10_STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def _unnormalize_cifar(img: np.ndarray) -> np.ndarray:
    mean = np.asarray(CIFAR_10_STATS[0], dtype=img.dtype).reshape(3, 1, 1)
    std = np.asarray(CIFAR_10_STATS[1], dtype=img.dtype).reshape(3, 1, 1)
    return img * std + mean


def show_images(dataloader, n_images=8, n_cols=4):
    images = []
    for batch in dataloader:
        images.extend(batch[0])
        if len(images) >= n_images:
            break
    images = images[:n_images]

    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i in range(n_images):
        image = images[i].cpu().numpy()
        image = _unnormalize_cifar(image)
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)  # Clip to valid range
        axes[i].imshow(image)
        axes[i].axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    return fig


class SoftmaxBrowser:
    def __init__(
        self, dataloader, model, class_names, with_noise=False, debouncing_time=50
    ) -> None:
        plt.close()

        images, _ = next(iter(dataloader))
        self.images = images
        self.model = model
        self.class_names = class_names
        self.with_noise = with_noise
        self.index = 0

        self.fig = plt.figure(layout="constrained", figsize=(9, 5))

        gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[1, 3])
        self.subfig_left = self.fig.add_subfigure(gs[0, 0])
        self.subfig_right = self.fig.add_subfigure(gs[0, 1])

        gs = gridspec.GridSpec(3, 2, height_ratios=[5, 1, 1], left=0.2)
        self.ax_img = self.subfig_left.add_subplot(gs[0, :])
        self.ax_slider = self.subfig_left.add_subplot(gs[1, :])
        self.ax_btn1 = self.subfig_left.add_subplot(gs[2, 0])
        self.ax_btn2 = self.subfig_left.add_subplot(gs[2, 1])

        [self.ax_logits, self.ax_preds] = self.subfig_right.subplots(2, 1, sharex=True)

        if self.with_noise:
            self.slider = Slider(self.ax_slider, "Noise", 0, 0.5, valinit=0)
            self.slider.on_changed(lambda _val: self.on_slider_move())
        else:
            self.ax_slider.axis("off")

        self.ax_img.axis("off")
        self.im = self.ax_img.imshow(
            _unnormalize_cifar(self.images[self.index].cpu().numpy()).transpose(1, 2, 0)
        )

        self.btn1 = Button(self.ax_btn1, "←")
        self.btn2 = Button(self.ax_btn2, "→")
        self.btn1.on_clicked(self.prev_image)
        self.btn2.on_clicked(self.next_image)

        self.timer = self.fig.canvas.new_timer(interval=debouncing_time)
        self.timer.add_callback(self.update_preds)

        self.update_image()
        self.update_preds()

    def current_image(self):
        img = self.images[self.index]
        if self.with_noise:
            noise_amp = self.slider.val
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.index)
            noise = torch.randn(img.shape, generator=gen)
            return (1 - noise_amp) * img + noise * noise_amp
        return img

    def on_slider_move(self):
        self.update_image()
        self.timer.stop()
        self.timer.start()

    def update_image(self):
        img = self.current_image().cpu().numpy()
        img = _unnormalize_cifar(img)
        img = img.transpose(1, 2, 0)
        self.im.set_array(img)
        self.fig.canvas.draw_idle()

    def update_preds(self):
        self.timer.stop()
        img = self.current_image()

        logits = self.model(img.unsqueeze(0))
        logits = logits.squeeze(0).detach().cpu().numpy()
        preds = np.exp(logits) / np.sum(np.exp(logits))

        self.ax_logits.clear()
        self.ax_logits.bar(np.arange(len(preds)), logits, color="lightcoral")
        self.ax_logits.grid(True, alpha=0.3, axis="y")
        self.ax_logits.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        self.ax_logits.set_ylabel("Logits")

        self.ax_preds.clear()
        self.ax_preds.bar(np.arange(len(preds)), preds, color="orange")
        self.ax_preds.grid(True, alpha=0.3, axis="y")
        self.ax_preds.set_ylabel("Probabilities")
        self.ax_preds.set_ylim(0, 1.0)
        self.ax_preds.set_xticks(range(len(preds)))
        self.ax_preds.set_xticklabels(self.class_names, rotation=45, ha="right")

        self.fig.canvas.draw_idle()

    def next_image(self, _event):
        self.index = (self.index + 1) % len(self.images)
        self.update_image()
        self.update_preds()

    def prev_image(self, _event):
        self.index = (self.index - 1) % len(self.images)
        self.update_image()
        self.update_preds()


def fast_dirichlet_pdf(p, alpha):
    alpha = np.array(alpha)
    b_alpha = np.prod(gamma(alpha)) / gamma(np.sum(alpha))
    prod_term = np.prod(p ** (alpha - 1), axis=1)
    return prod_term / b_alpha


def barycentric_to_cartesian(b1, b2, b3):
    """Convert barycentric coords to 2D cartesian for this triangle."""
    corners = np.array([[0, 0], [1, 0], [0.5, 3**0.5 / 2]])
    return b1 * corners[0] + b2 * corners[1] + b3 * corners[2]


def draw_ticks_and_grid(ax, n_ticks=5, tick_len=0.025, grid_alpha=0.2):
    """Draw barycentric tick marks and optional grid lines on the triangle."""
    corners = np.array([[0, 0], [1, 0], [0.5, 3**0.5 / 2]])
    tick_values = np.linspace(0, 1, n_ticks + 2)[1:-1]  # e.g. 0.2, 0.4, 0.6, 0.8

    # Inward normal directions for each edge (perpendicular, pointing inward)
    edge_normals = [
        np.array([0, 1]),  # bottom edge (p1-p2): normal points up
        np.array([-(3**0.5) / 2, 0.5]),  # right edge (p2-p3): normal points left-up
        np.array([3**0.5 / 2, 0.5]),  # left edge (p3-p1): normal points right-up
    ]

    # Each edge: parameterized from corner[i] to corner[j]
    edges = [(0, 1), (1, 2), (2, 0)]

    for (i, j), normal in zip(edges, edge_normals):
        c0, c1 = corners[i], corners[j]
        for t in tick_values:
            pt = (1 - t) * c0 + t * c1
            tick_start = pt
            tick_end = pt + tick_len * normal
            ax.plot(
                [tick_start[0], tick_end[0]],
                [tick_start[1], tick_end[1]],
                color="black",
                linewidth=0.8,
                zorder=5,
            )

            # Tick label: value along this edge (as fraction of the *opposite* axis)
            label_pos = pt - tick_len * 1.8 * normal
            ax.text(
                label_pos[0],
                label_pos[1],
                f"{t:.1f}",
                fontsize=7,
                ha="center",
                va="center",
                color="gray",
            )

    # Draw interior grid lines (iso-lines of each barycentric coordinate)
    for t in tick_values:
        for coord_idx in range(3):
            # Two points where barycentric coord `coord_idx` == t
            pts = []
            for e0, e1 in [(0, 1), (1, 2), (2, 0)]:
                # Parameterize the edge from corner e0 to corner e1
                # Barycentric coords of the endpoints
                bc0 = np.zeros(3)
                bc0[e0] = 1
                bc1 = np.zeros(3)
                bc1[e1] = 1
                # Find s such that (1-s)*bc0[coord_idx] + s*bc1[coord_idx] == t
                d = bc1[coord_idx] - bc0[coord_idx]
                if abs(d) > 1e-10:
                    s = (t - bc0[coord_idx]) / d
                    if 0 <= s <= 1:
                        pt = (1 - s) * corners[e0] + s * corners[e1]
                        pts.append(pt)
            if len(pts) == 2:
                ax.plot(
                    [pts[0][0], pts[1][0]],
                    [pts[0][1], pts[1][1]],
                    color="white",
                    linewidth=0.5,
                    alpha=grid_alpha,
                    zorder=4,
                    linestyle="--",
                )


class InteractiveDirichlet:
    """
    Interactive Matplotlib visualization for a 3-class Dirichlet distribution.
    """

    def __init__(
        self, init_alpha: Optional[List[float]] = None, max_alpha: float = 10.0
    ):
        plt.close()

        if init_alpha is None:
            init_alpha = [2.0, 2.0, 2.0]

        self.max_alpha = max_alpha
        self._setup_mesh()
        self._setup_plot()
        self._setup_sliders(init_alpha)
        self.update(None)

    def _setup_mesh(self):
        """Creates the barycentric coordinate mesh for the simplex."""
        triangle_height = np.sqrt(0.75)
        corners = np.array([[0, 0], [1, 0], [0.5, triangle_height]])

        self.triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        refiner = tri.UniformTriRefiner(self.triangle)
        self.trimesh = refiner.refine_triangulation(subdiv=5)

        self.x, self.y = self.trimesh.x, self.trimesh.y

        # Convert Cartesian to Barycentric coordinates
        p3 = self.y / triangle_height
        p2 = self.x - 0.5 * p3
        p1 = 1.0 - p2 - p3

        p_raw = np.stack((p1, p2, p3), axis=-1)
        self.p = np.clip(p_raw, 1e-12, 1.0)

    def _setup_plot(self):
        """Initializes the matplotlib figure, axes, and static visuals."""
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.35, right=0.85)
        self.cbar_ax = self.fig.add_axes((0.88, 0.4, 0.03, 0.45))

        self.ax.axis("equal")
        self.ax.axis("off")

        labels = {"class 1": (0.1, 0.4), "class 2": (0.5, -0.14), "class 3": (0.9, 0.4)}
        for text, (x, y) in labels.items():
            self.ax.text(x, y, text, fontsize=13, ha="center")

        draw_ticks_and_grid(self.ax, n_ticks=4, tick_len=0.025, grid_alpha=0.25)

        self.contour_plot = None

    def _setup_sliders(self, init_alpha: List[float]):
        """Creates the interactive sliders for the alpha parameters."""
        axcolor = "lightgray"

        # Store axes as instance variables in case they need to be accessed later
        self.ax_a1 = plt.axes((0.15, 0.20, 0.65, 0.03), facecolor=axcolor)
        self.ax_a2 = plt.axes((0.15, 0.13, 0.65, 0.03), facecolor=axcolor)
        self.ax_a3 = plt.axes((0.15, 0.06, 0.65, 0.03), facecolor=axcolor)

        self.slider_a1 = Slider(
            self.ax_a1, "α₁", 1.01, self.max_alpha, valinit=init_alpha[0]
        )
        self.slider_a2 = Slider(
            self.ax_a2, "α₂", 1.01, self.max_alpha, valinit=init_alpha[1]
        )
        self.slider_a3 = Slider(
            self.ax_a3, "α₃", 1.01, self.max_alpha, valinit=init_alpha[2]
        )

        self.slider_a1.on_changed(self.update)
        self.slider_a2.on_changed(self.update)
        self.slider_a3.on_changed(self.update)

    def update(self, val: Optional[float]):
        """Callback to redraw the PDF contour when sliders change."""
        alphas = [self.slider_a1.val, self.slider_a2.val, self.slider_a3.val]

        # Calculate PDF
        pdf_values = fast_dirichlet_pdf(self.p, alphas)

        # Clear previous contours
        for c in self.ax.collections:
            c.remove()

        self.contour_plot = self.ax.tricontourf(
            self.trimesh, pdf_values, cmap="viridis", levels=20
        )

        self.cbar_ax.cla()
        self.fig.colorbar(self.contour_plot, cax=self.cbar_ax)
        self.fig.canvas.draw_idle()


def compute_edl_metrics(evidence):
    evidence = evidence.detach().float().cpu().numpy()
    alpha = evidence + 1.0
    num_classes = alpha.shape[0]
    strength = float(np.sum(alpha))
    probs = alpha / strength

    vacuity = num_classes / strength
    predictive_entropy = -np.sum(probs * np.log(probs + 1e-10))
    expected_entropy = -np.sum(
        (alpha / strength) * (psi(alpha + 1.0) - psi(strength + 1.0))
    )
    mutual_information = predictive_entropy - expected_entropy
    variance = probs * (1.0 - probs) / (strength + 1.0)

    return {
        "evidence": evidence,
        "probs": probs,
        "vacuity": vacuity,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variance": variance,
    }


class EDLBrowser:
    def __init__(
        self, loader, model, class_names, with_noise=False, debouncing_time=50
    ) -> None:
        plt.close()
        images, _ = next(iter(loader))

        self.images = images
        self.model = model.to("cpu").eval()
        self.class_names = class_names
        self.with_noise = with_noise
        self.index = 0

        self.fig = plt.figure(layout="constrained", figsize=(9, 5))
        gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[1, 3])
        self.subfig_left = self.fig.add_subfigure(gs[0, 0])
        self.subfig_right = self.fig.add_subfigure(gs[0, 1])

        gs = gridspec.GridSpec(3, 2, height_ratios=[5, 1, 1], left=0.2)
        self.ax_img = self.subfig_left.add_subplot(gs[0, :])
        self.ax_slider = self.subfig_left.add_subplot(gs[1, :])
        self.ax_btn1 = self.subfig_left.add_subplot(gs[2, 0])
        self.ax_btn2 = self.subfig_left.add_subplot(gs[2, 1])

        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.2, bottom=0.2)
        self.ax_unc = self.subfig_right.add_subplot(gs[0, 0])
        self.ax_evidence = self.subfig_right.add_subplot(gs[1, 0])
        self.ax_preds = self.ax_evidence.twinx()

        if self.with_noise:
            self.slider = Slider(self.ax_slider, "Noise", 0, 0.5, valinit=0)
            self.slider.on_changed(lambda _val: self.on_slider_move())
        else:
            self.ax_slider.axis("off")

        self.ax_img.axis("off")
        self.im = self.ax_img.imshow(
            _unnormalize_cifar(self.images[self.index].cpu().numpy()).transpose(1, 2, 0)
        )

        self.btn1 = Button(self.ax_btn1, "←")
        self.btn2 = Button(self.ax_btn2, "→")
        self.btn1.on_clicked(self.prev_image)
        self.btn2.on_clicked(self.next_image)

        self.timer = self.fig.canvas.new_timer(interval=debouncing_time)
        self.timer.add_callback(self.update_preds)

        self.update_image()
        self.update_preds()

    def on_slider_move(self):
        self.update_image()
        self.timer.stop()
        self.timer.start()

    def current_image(self):
        img = self.images[self.index]
        if self.with_noise:
            noise_amp = self.slider.val
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.index)
            noise = torch.randn(img.shape, generator=gen)
            return (1 - noise_amp) * img + noise * noise_amp
        return img

    def update_image(self):
        img = self.current_image()
        img = _unnormalize_cifar(img.cpu().numpy())
        img = img.transpose(1, 2, 0)
        self.im.set_array(img)
        self.fig.canvas.draw_idle()

    def update_preds(self):
        self.timer.stop()
        img = self.current_image()

        evidence = self.model(img.unsqueeze(0).to("cpu"))
        metrics = compute_edl_metrics(evidence.squeeze(0))
        evidence = metrics["evidence"]
        preds = metrics["probs"]

        self.ax_evidence.clear()
        self.ax_preds.clear()

        self.ax_evidence.bar(
            np.arange(len(preds)) - 0.15, evidence, 0.25, color="lightcoral"
        )
        self.ax_evidence.set_ylabel("Evidence", color="lightcoral", fontweight="bold")
        self.ax_evidence.set_xticks(range(len(preds)))
        self.ax_evidence.set_xticklabels(self.class_names, rotation=45, ha="right")
        self.ax_evidence.set_ylim(0, max(evidence.max() * 1.2, 3))

        self.ax_preds.bar(np.arange(len(preds)) + 0.15, preds, 0.25, color="orange")
        self.ax_preds.yaxis.set_label_position("right")
        self.ax_preds.set_ylabel("Probabilities", color="orange", fontweight="bold")
        self.ax_preds.set_ylim(0, 1.05)

        self.ax_unc.clear()
        self.ax_unc.axis("off")
        metrics_text = (
            f"Predictive Entropy (total.): {metrics['predictive_entropy']:.3f}\n"
            f"Vacuity (ep.): {metrics['vacuity']:.3f}\n"
            f"Mutual Information (ep.): {metrics['mutual_information']:.3f}\n"
            f"Exp. Variance (ep.): {metrics['variance'].max():.3f}\n"
            f"Exp. Entropy (al.): {metrics['expected_entropy']:.3f}\n"
        )
        self.ax_unc.text(
            0.05,
            0.95,
            metrics_text,
            transform=self.ax_unc.transAxes,
            verticalalignment="top",
            family="monospace",
            fontsize=10,
            linespacing=1.5,
        )

        self.fig.canvas.draw_idle()

    def next_image(self, _event):
        self.index = (self.index + 1) % len(self.images)
        self.update_image()
        self.update_preds()

    def prev_image(self, _event):
        self.index = (self.index - 1) % len(self.images)
        self.update_image()
        self.update_preds()
