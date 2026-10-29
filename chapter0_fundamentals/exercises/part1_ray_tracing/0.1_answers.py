# %%
#setup 

import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%
# Exercise - implement make_rays_1d
from torch import Tensor

def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """

    result = t.empty(num_pixels, 2, 3)
    result[:, 0, :3] = 0
    result[:, 1, :3] = 1
    result[:, 1, 0] = 1    
    result[:, 1, 1] = t.linspace(start=-y_limit, end=y_limit, steps=num_pixels)

    result[:, 1, 2] = 0

    print(result)
    return result


tests.test_make_rays_1d(make_rays_1d)

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# %%
#

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """

    A = t.stack((ray[1], segment[0] - segment[1]), dim=1)[:2, :]
    assert A.shape == (2, 2), f"Expected shape (2, 2), got {A.shape}"

    B = (segment[0] - ray[0])[:2]
    assert B.shape == (2,), f"Expected shape (2), got {B.shape}"

    try:
        uv = t.linalg.solve(A, B)
    except t.linalg.LinAlgError:
        return False
    
    assert uv.shape == (2,)
    if uv[0] >= 0 and uv[1] <= 1 and uv[1] >= 0:
        return True
    else:
        return False


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%
#

def intersect_rays_1d(
    rays: Float[Tensor, "num_rays 2 3"], segments: Float[Tensor, "num_segments 2 3"]
) -> Bool[Tensor, "num_rays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    num_rays = rays.shape[0]
    num_segments = segments.shape[0]

    origins_3d = rays[:,0]
    assert origins_3d.shape == (num_rays,3)
    D_3D = rays[:,1]
    print(D_3D.shape)
    assert D_3D.shape == (num_rays,3)

    L1_3d = segments[:,0]
    assert L1_3d.shape == (num_segments,3)
    L2_3d = segments[:,1]
    assert L2_3d.shape == (num_segments,3)
    Ldiff_3d = L1_3d - L2_3d
    assert Ldiff_3d.shape == (num_segments,3)

    D_2D = D_3D[:,:2]
    assert D_2D.shape == (num_rays,2), f"Expected shape ({num_rays}, 2), got {D_2D.shape}"
    O_2D: Float[Tensor, "2"] = origins_3d[:,:2]
    assert O_2D.shape == (num_rays,2), f"Expected shape ({num_rays}, 2), got {O_2D.shape}"
    L1_2D: Float[Tensor, "num_segments 2"] = L1_3d[:,:2]
    assert L1_2D.shape == (num_segments,2), f"Expected shape ({num_segments}, 2), got {L1_2D.shape}"
    Ldiff_2D = Ldiff_3d[:,:2]
    assert Ldiff_2D.shape == (num_segments,2), f"Expected shape ({num_segments}, 2), got {Ldiff_2D.shape}"

    Dxy: Float[Tensor, "num_rays num_segments 2"] = einops.repeat(D_2D, "num_rays d -> num_rays num_segments d", num_segments = num_segments)
    Ldiffxy: Float[Tensor, "num_rays num_segments 2"] = einops.repeat(Ldiff_2D, "num_segments d -> num_rays num_segments d", num_rays=num_rays)
    assert Dxy.shape == Ldiffxy.shape, f"Expected shape (Dxy.shape), got {Ldiffxy.shape}"

    A = t.stack((Dxy, Ldiffxy), dim=-1)
    assert A.shape == (num_rays, num_segments, 2, 2), f"Expected shape (num_rays, num_segments, 2, 2), got {A.shape}"


    # Target vector B = (L1_2D - O_2D) -> shape(num_rays, num_segments, 2)
    # L1_2D: (1, num_segments, 2), O_2D: (num_rays, 1, 2)
    B = L1_2D[:, :2].unsqueeze(0) - O_2D[:, :2].unsqueeze(1)
    assert B.shape == (num_rays, num_segments, 2), f"Expected shape (num_rays, num_segments, 2), got {B.shape}"

    dets = t.linalg.det(A)
    assert dets.shape == (num_rays, num_segments)
    is_invertible: Bool[Tensor, "num_rays num_segments"] = t.abs(dets) > 1e-6
    assert is_invertible.shape == (num_rays, num_segments) 

    uv: Float[Tensor, "num_rays num_segments 2"] = t.zeros((num_rays, num_segments, 2), dtype=t.float32)

    uv[is_invertible] = t.linalg.solve(A[is_invertible], B[is_invertible])

    # Check intersection conditions: u >= 0 and 0 <= v <= 1
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    intersections: Bool[Tensor, "num_rays num_segments"] = is_invertible & (u >= 0) & (v >= 0) & (v <= 1) 

    result: Bool[Tensor, "num_rays"] = intersections.any(dim=1)

    return result


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
