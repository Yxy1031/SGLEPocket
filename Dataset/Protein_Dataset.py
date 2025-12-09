import torch
import torch.nn.functional as F
from skimage.morphology import binary_dilation
from skimage.morphology import cube
from .MolDatasets import MolDataset
import molgrid
import numpy as np


def get_mask(coordinateset, center, gmaker):
    # Create ground truth tensor
    c2grid = molgrid.Coords2Grid(gmaker, center=center)
    origtypes = torch.ones(coordinateset.coords.tonumpy().shape[0], 1)
    radii = torch.ones((coordinateset.coords.tonumpy().shape[0]))
    grid_gen = c2grid(torch.tensor(coordinateset.coords.tonumpy()), origtypes, radii)
    grid_np = grid_gen.numpy()
    grid_np = binary_dilation(grid_np[0], cube(3))
    grid_np = grid_np.astype(float)
    return torch.tensor(np.expand_dims(grid_np, axis=0))



class TrainScPDB(MolDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_num = 400
        self.atomtype = 14
        self.max_dist = 32
        self.resolution = 1
        self.size = int(2 * self.max_dist/self.resolution) + 1
        self.gmaker_mask = molgrid.GridMaker(dimension=32, binary=True, gaussian_radius_multiple=-1, resolution=0.5)
        self.gmaker_img = molgrid.GridMaker(dimension=32, radius_scale=1.0)

    def __getitem__(self, item):
        center, coords, atomtypes, radii, labels, coord_sets = super(TrainScPDB, self).__getitem__(item)
        pocket = coord_sets[-1]
        protein = coord_sets[0]
        input_tensor = torch.zeros([14, self.size, self.size, self.size], dtype=torch.float32)
        mask_tensor = torch.zeros([1, self.size, self.size, self.size], dtype=torch.float32)
        protein_coords = torch.tensor(protein.coords.tonumpy())
        prontein_atomtypes = torch.tensor(protein.type_index.tonumpy())
        protein_radii = torch.tensor(protein.radii.tonumpy())
        protein_coords, prontein_atomtypes, protein_radii = self.knn(protein_coords, torch.tensor(labels[1:]), prontein_atomtypes, protein_radii)
        # protein_ = molgrid.CoordinateSet(protein_coords.numpy(), prontein_atomtypes.numpy(), protein_radii.numpy(), self.atomtype)
        centers = molgrid.float3(float(labels[1]), float(labels[2]), float(labels[3]))
        self.gmaker_img.forward(centers, protein, input_tensor)


        mask_tensor = get_mask(pocket, centers, self.gmaker_mask)
        return input_tensor, mask_tensor, list([float(labels[1]), float(labels[2]), float(labels[3])]), protein.src,protein_coords



    def knn(self, Points, xyz, atomtypes, radii):
        num = self.points_num
        ref_c = torch.stack([xyz] * Points.shape[0], dim=0)
        query_c = Points
        delta = query_c - ref_c
        distances = torch.sqrt(torch.pow(delta, 2).sum(dim=1))
        sorted_dist, indices = torch.sort(distances)
        # if sorted_dist[num] > 16:
        #     print("error!! num out of edge")
        #num = torch.where(sorted_dist>16)[0][0]
        return query_c[indices[:num]], atomtypes[indices[:num]], radii[indices[:num]]

    def filter_points(self,Points, xyz, atomtypes, radii, distance_threshold= 16):
        distances = np.abs(Points - xyz)
        mask = torch.all(distances <= distance_threshold, dim=1)
        return Points[mask] ,atomtypes[mask], radii[mask]


