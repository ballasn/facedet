# ------- Detection parameters ------- #
# ------- Called by launchAFW  ------- #

# Models
models = ['../cascade/models/dsn48large_sig_best_fine_unfreeze.pkl']
models.append('../cascade/models/modele96_conv.pkl')
# Sizes of the inputs
sizes = [48, 96]
# Strides of the inputs, used to cut the images in pieces
strides = [8, 16]

# Scales
from math import sqrt
ratio = sqrt(2)
global_scales = [(1.0/ratio)**e for e in range(-3, 10)]
global_scales2 = [0.9, 1.0, 1.1]
cascade_scales = [global_scales, global_scales2]

# Parameters
# Threshold for each stage
probs = [-1.0, -1.0]
# Overlap allowed at each stage
overlap_ratio = [0.5, 0.3]
# Minimal size of a prediction
min_pred_size = 30
# Size of the pieces when cutting the image
piece_size = 300
# Maximal size of the input image, resizing would happen before cascade
# function
max_img_size = float(700)
