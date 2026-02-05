import datetime
import torch
import os
import argparse
from torch import nn
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from os.path import join
import molgrid
from skimage.morphology import closing
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from SGLEPocket.SGLEPocketnet import SGLEPocket
from openbabel import pybel
import prody
from Dataset import Protein_Dataset

pybel.ob.obErrorLog.SetOutputLevel(0)
prody.confProDy(verbosity='none')

def get_model_gmaker_eprovider(test_types, batch_size, data_dir, dims=None):
    """Load data"""
    eptest_large = Protein_Dataset.TrainScPDB(test_types, data_root=data_dir, cache_structs=True)
    eptest_large.set_transform(False)
    test_loader = torch.utils.data.DataLoader(eptest_large,
                                             batch_size=batch_size,
                                             pin_memory=True)
    if dims is None:
        gmaker = molgrid.GridMaker()
    else:
        gmaker = molgrid.GridMaker(dimension=dims)
    return gmaker, test_loader

def preprocess_output_3d_solid(input_tensor, threshold):
    """Post-processing for 3D solid volume"""
    if torch.is_tensor(input_tensor):
        data = input_tensor.cpu().numpy()
    else:
        data = input_tensor

    binary_data = (data >= threshold).astype(np.int8)

    if np.sum(binary_data) == 0:
        print("  Warning: Prediction result is empty")
        return torch.tensor(binary_data, dtype=torch.float32)

    label_image, num_labels = label(binary_data, return_num=True)

    if num_labels == 0:
        return torch.tensor(binary_data, dtype=torch.float32)

    sizes = np.bincount(label_image.ravel())
    largest_label = sizes[1:].argmax() + 1
    largest_mask = (label_image == largest_label)

    filled_mask = binary_fill_holes(largest_mask).astype(np.float32)

    print(f"  Voxel count: {np.sum(filled_mask)}")

    return torch.tensor(filled_mask, dtype=torch.float32)

def Output_Coordinates(tensor, center, dimension=16.25, resolution=0.5):
    """Output coordinates"""
    tensor = tensor.numpy()
    indices = np.argwhere(tensor > 0).astype('float32')
    indices *= resolution
    center = np.array([float(center[0]), float(center[1]), float(center[2])])
    indices += center
    indices -= dimension
    return indices

def save_density_as_cube(density, origin, step, fname, name='pocket_site'):
    """Save voxel data in Gaussian Cube format"""
    angstrom2bohr = 1.889725989

    if len(density.shape) > 3:
        density = np.squeeze(density)

    os.makedirs(os.path.dirname(fname) if os.path.dirname(fname) else '.', exist_ok=True)

    with open(fname, 'w') as f:
        f.write('%s CUBE FILE\n' % name)
        f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
        f.write(' 1 %12.6f %12.6f %12.6f\n' % tuple(angstrom2bohr * np.array(origin)))

        f.write(
            '%5i %12.6f 0.000000 0.000000\n'
            '%5i 0.000000 %12.6f 0.000000\n'
            '%5i 0.000000 0.000000 %12.6f\n' % (
                density.shape[0], angstrom2bohr * step[0],
                density.shape[1], angstrom2bohr * step[1],
                density.shape[2], angstrom2bohr * step[2]
            )
        )

        f.write(' 1 0.000000 %12.6f %12.6f %12.6f\n' % tuple(angstrom2bohr * np.array(origin)))

        flat_data = density.flatten()
        for i in range(0, len(flat_data), 6):
            chunk = flat_data[i:i+6]
            f.write(" ".join(["%12.6f" % v for v in chunk]) + "\n")

    print(f"  Saved: {fname}")

def run_prediction(args):
    """Run prediction"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Pocket Prediction")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Types File: {args.types_file}")
    print(f"Data Root: {args.data_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = SGLEPocket(
        spatial_dims=3,
        init_filters=24,
        in_channels=14,
        out_channels=1,
        dropout_prob=0.2,
        blocks_down=(1, 1, 1),
        blocks_bottleneck=2,
        blocks_up=(1, 1, 1),
        use_LFE=True,
    )
    model.to(device)
    model = nn.DataParallel(model)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded.\n")

    # Load data
    print("Loading data...")
    gmaker, test_loader = get_model_gmaker_eprovider(
        args.types_file,
        batch_size=1,
        data_dir=args.data_dir,
        dims=32
    )
    print(f"Data loaded, total {len(test_loader)} samples.\n")

    types_name = os.path.basename(args.types_file).replace('.types', '')

    print(f"{'='*60}")
    print("Starting prediction...")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"[Sample {i+1}/{len(test_loader)}] {types_name}")

            input_tensor, mask_tensor, center, labels, protein_coords = batch

            masks_pred = model(input_tensor.to(device)[:, :14])

            # Post-processing
            pred = masks_pred
            pred2 = pred
            edge3 = torch.zeros_like(pred)
            edge_fuse = 0.3*edge3 + 1*pred2 + 1.5*edge3*pred

            masks_pred = edge_fuse.detach().cpu()

            processed_mask = preprocess_output_3d_solid(masks_pred[0].squeeze(), args.threshold)

            # Calculate coordinates
            f_center = np.array([center[0].item(), center[1].item(), center[2].item()])
            pred_coords = Output_Coordinates(processed_mask, center)

            if len(pred_coords) > 0:
                pred_centroid = pred_coords.mean(axis=0)
                print(f"  Predicted Centroid: [{pred_centroid[0]:.2f}, {pred_centroid[1]:.2f}, {pred_centroid[2]:.2f}]")
                print(f"  Predicted Points: {len(pred_coords)}")
            else:
                print(f"  Warning: No valid prediction")

            # Save Cube file
            res = 1.0 / args.scale
            step = np.array([res, res, res])
            origin = f_center - args.max_dist

            output_filename = f"{types_name}_pocket_{i+1}.cube"
            cube_path = join(args.output_dir, output_filename)
            save_density_as_cube(
                processed_mask.numpy(),
                origin,
                step,
                cube_path,
                name=f'{types_name}_pocket'
            )

            print()

    print(f"{'='*60}")
    print(f"Prediction finished! Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SGLEPocket Pocket Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model weight file')
    parser.add_argument('--types_file', type=str, required=True,
                        help='Path to .types file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data root directory (usually where test_types resides)')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory (Default: ./predictions)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (Default: 0.5)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID (Default: 0)')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='Voxel resolution scale factor (Default: 2.0)')
    parser.add_argument('--max_dist', type=float, default=16.25,
                        help='Max distance parameter')


    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        exit(1)

    if not os.path.exists(args.types_file):
        print(f"Error: Types file not found: {args.types_file}")
        exit(1)

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run_prediction(args)