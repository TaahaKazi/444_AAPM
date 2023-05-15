"""Date    : Jan 30th, 2023

(c) AAPM DGM-Image Challenge Organizers.

This source-code is licensed under the terms outlined in the accompanying LICENSE file.
"""

import argparse
import numpy as np
from scipy import stats
import glob
import os
import os.path as p
import imageio as io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib
import cv2

matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('use_inf_as_na', True)  # All infs throughout are set to nans


def load_data(path, num_images, convert_size=512, convert=False):
    data = np.zeros((num_images, convert_size, convert_size), dtype=np.uint8)

    fnames = sorted(glob.glob(p.join(path, '*.png')))[:num_images]
    for i, f in enumerate(fnames):
        print(f"Loading data {i}/{num_images}", end='\r')
        if convert:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (convert_size, convert_size), interpolation=cv2.INTER_AREA)
            data[i] = resized
            pass
        else:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if (len(img.shape)==3): # converting rgb to b/w
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            data[i] = img
    data = data.astype(int)
    return fnames, data


def public_features(data):
    data = np.squeeze(data).astype(np.float32)

    phantom_masks = (data > 45).astype(float)

    print("Computing slice areas")
    areas = np.sum(phantom_masks, axis=(1, 2))

    print("Computing fat area")
    fat = (data >= 45) * (data < 120)
    fat_areas = np.sum(fat, axis=(1, 2))

    print("Computing glandular area")
    gln = (data >= 120) * (data < 226)
    gln_areas = np.sum(gln, axis=(1, 2))

    print("Computing fat to glandular ratio")
    fg_ratios = np.log10(fat_areas / gln_areas)

    N = len(data)
    data = data.reshape(N, -1)

    print("Computing means ...")
    means = np.mean(data, axis=1)

    print("Computing stds ...")
    stds = np.std(data, axis=1)

    print("Computing skewnesses ...")
    skewnesses = stats.skew(data, axis=1)

    print("Computing kurtoses ...")
    kurtoses = stats.kurtosis(data, axis=1)

    print("Computing balances")
    balances = (np.quantile(data, 0.7, axis=1) - np.mean(data, axis=1)) \
               / (np.mean(data, axis=1) - np.quantile(data, 0.3, axis=1))

    return {'mean': means,
            'std': stds,
            'skewness': skewnesses,
            'kurtosis': kurtoses,
            'balance': balances,

            'area': areas,
            'fat_area': fat_areas,
            'gln_area': gln_areas,
            'fg_ratio': fg_ratios,
            }


parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--results_dir", type=str, default='metric_results', help="Results directory")

# data arguments
parser.add_argument("--data_name", type=str, default='downsized_victre_xray_objects2',
                    help="Data name (innermost folder name used for stylegan training)")
parser.add_argument("--path_to_reals", type=str, default='',
                    help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--path_to_fakes", type=str, default='',
                    help="Path to the folder containing the .png images (real or fake)")
parser.add_argument("--num_images", type=int, default=1000, help="Number of images to load")
parser.add_argument("--levels", type=int, default=64, help='Number of levels used for image digitization')

# summary arguments
parser.add_argument("--num_random_runs", default=100000, type=int,
                    help='Number of samples to draw from while measuring the cosine distance.')
parser.add_argument("--pca_components", default=9, type=int, help='Number of PCA components to use.')
parser.add_argument("--to_save_pca_plots", default=1, type=lambda b: bool(int(b)), help="save pca plots")
parser.add_argument("--to_save_cosine_plots", default=0, type=lambda b: bool(int(b)),
                    help="save cosine plots. Requires seaborn installed.")
args = parser.parse_args()

if args.results_dir != '':
    os.makedirs(args.results_dir, exist_ok=True)

# load real or fake data
real_data = {}
fnames, imgs = load_data(args.path_to_reals, args.num_images, convert_size=256, convert=True)
real_data['fnames'] = fnames
real_data['data'] = imgs
print("Real data loaded")

fake_data = {}
fnames, imgs = load_data(args.path_to_fakes, args.num_images, convert_size=256)
fake_data['fnames'] = fnames
fake_data['data'] = imgs

# Evaluate the public metrics
real_features = public_features(real_data['data'])
fake_features = public_features(fake_data['data'])

real_df = pd.DataFrame.from_dict(real_features).fillna(0)
fake_df = pd.DataFrame.from_dict(fake_features).fillna(0)

print("Computing real PCA")
# PCA the reals and store the PCA'd values in a dataframe
scalar = StandardScaler()
real_np = real_df.to_numpy()
scalar.fit(real_np)
R_scaled = scalar.transform(real_np)
pca = PCA(n_components=args.pca_components, random_state=1000)
R_pca = pca.fit_transform(R_scaled)
dfr_pca = pd.DataFrame(R_pca)

print("Projecting fake data onto the real PC components")
fake_np = fake_df.to_numpy()
F_scaled = scalar.transform(fake_np)
F_pca = pca.transform(F_scaled)
dff_pca = pd.DataFrame(F_pca)

# Compute pairwise cosine distances for the reals
print("Computing pairwise cosine distances for the reals")
real_cosines = []
for i in range(args.num_random_runs):
    print(f'{i}/{args.num_random_runs}', end='\r')

    sampled = dfr_pca.sample(n=2, replace=False).to_numpy()
    real_cosines.append(
        np.dot(sampled[0, :], sampled[1, :]) / (np.linalg.norm(sampled[0, :]) * np.linalg.norm(sampled[1, :])))

# Compute pairwise cosine distances for real-fake point pairs
print("Computing pairwise cosine distances for real-fake point pairs")
mixed_cosines = []
for i in np.arange(args.num_random_runs):
    sampledr = dfr_pca.sample(n=1, replace=False).to_numpy()
    sampledf = dff_pca.sample(n=1, replace=False).to_numpy()
    mixed_cosines.append(
        np.dot(sampledr[0, :], sampledf[0, :]) / (np.linalg.norm(sampledr[0, :]) * np.linalg.norm(sampledf[0, :])))

# Compute KS statistic on the reals and the mixed distributions of cosine similarities.
print("Compute KS statistic")
ks_stat, _ = ks_2samp(real_cosines,
                      mixed_cosines)  # This will slightly vary over multiple runs, probably in the second decimal place.
print("KS statistic : ", ks_stat)
ks_stat_str = f'{ks_stat:.3f}'

if args.to_save_pca_plots:
    print("Saving PCA plots")
    plt.figure()
    plt.scatter(F_pca[:1000, 0], F_pca[:1000, 1], color='red', alpha=0.2, label='fake')
    plt.scatter(R_pca[:1000, 0], R_pca[:1000, 1], color='green', alpha=0.2, label='real')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA')
    plt.legend()
    plt.savefig(p.join(args.results_dir, f'crop_20pca.png'))
    plt.close()

if args.to_save_cosine_plots:
    import seaborn as sns

    print("Saving cosine plots")
    plt.figure()
    sns.kdeplot(np.array(real_cosines), label='real')
    sns.kdeplot(np.array(mixed_cosines), label='mixed')
    plt.ylabel('density')
    plt.xlabel('cosine similarity')
    plt.title(f'KS statistic : {ks_stat_str}')
    plt.legend()
    plt.savefig(p.join(args.results_dir, f'10cosine.png'))
    plt.close()

with open(p.join(args.results_dir, f'public_metric.txt'), 'w') as fid:
    fid.write(f"KS statistic   : {ks_stat:.6f}")
