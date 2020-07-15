# Use imageio and pathlib to convert from a list of png frames
# to gifs
import pathlib

import imageio
import pygifsicle

# BASE_DIR  = '/home/mike/research/ac6_curtains/all_sky_imagers/plots/movies/log/ac6_alt'
BASE_DIR  = '/home/mike/research/ac6_curtains/all_sky_imagers/plots/movies/log/100km_alt'


# List off all the date folders (rglob will recursively return the png files inside.)
day_dirs_generator = pathlib.Path(BASE_DIR).glob('*') 
day_dirs = sorted(list(day_dirs_generator))

for day_dir in day_dirs:
    # First get the paths to all the png files in each day sub-directory.
    img_paths = sorted(list(day_dir.glob('*.png')))

    # Now use imagio to write a gif movie.
    gif_path = day_dir.parent / f'{day_dir.name}_movie.gif'

    with imageio.get_writer(gif_path, mode='I') as writer:
        for img in img_paths:
            writer.append_data(imageio.imread(img))

    # Now optimize the gif by saving only the differences
    # between the frames.
    pygifsicle.optimize(str(gif_path))