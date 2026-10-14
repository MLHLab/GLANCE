================================================================================
GLANCE: A Federated, CBAM-based Learning Framework for Cross-Country
Multi-Class Classification of Lung Infections
Code README
================================================================================

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This package contains the Python scripts used to:

  (a) train a CBAM-based convolutional neural network (CNN) on the chest X-ray
      data of one client (country) at a time (USA, China, India);
  (b) aggregate the three client models at a central server node using
      federated averaging (unweighted mean of the model weights); and
  (c) train four baseline CNN architectures (DenseNet, InceptionV3, ResNet50,
      VGG16) on the pooled data, for comparison with the proposed classifier.

Task: three-class chest X-ray classification (Normal, Tuberculosis, Pneumonia).
Input size: 180 x 180 x 3, pixel values rescaled to [0, 1].


--------------------------------------------------------------------------------
2. FILES
--------------------------------------------------------------------------------
  ClientNode.py              Trains the proposed CBAM-based classifier on one
                             client's local images and saves its weights.
  ServerNode.py              Loads the three client weight files and writes the
                             averaged (central) weights.
  Full_Data_DenseNet.py      Baseline: DenseNet-style network (built from
                             scratch), trained on the pooled data.
  Full_Data_InceptionV3.py   Baseline: InceptionV3 (truncated at 'mixed7').
  Full_Data_ResNet50.py      Baseline: ResNet50 (ImageNet weights).
  Full_Data_VGG16.py         Baseline: VGG16 (ImageNet weights).

Client numbering used by the scripts (see comment in ServerNode.py):
  Client1 = USA,   Client2 = China,   Client3 = India


--------------------------------------------------------------------------------
3. REQUIREMENTS
--------------------------------------------------------------------------------
  Python 3            [version used: to be filled in]
  numpy, scipy, pandas, matplotlib, Pillow
  tensorflow / keras  [versions used: to be filled in]
  scikit-learn
  Jupyter / IPython   (see note below)
  Standard library: pathlib, pickle, datetime, collections, os, random

Notes:
  - The scripts are exports of Jupyter notebooks (cells are marked "# In[n]:").
    ClientNode.py and Full_Data_ResNet50.py call get_ipython(), so they must be
    run inside Jupyter/IPython (or the get_ipython() line must be removed).
  - The scripts import ImageDataGenerator from the legacy keras.preprocessing
    API, so a TensorFlow/Keras installation that still provides it is needed.
  - Full_Data_ResNet50.py (weights='imagenet') and Full_Data_VGG16.py (default
    weights) download ImageNet weights on first use; an internet connection
    is required for that step.
  - A GPU is optional; no hardware-specific code is used.


--------------------------------------------------------------------------------
4. DATA ORGANISATION
--------------------------------------------------------------------------------
Images are read with flow_from_directory, so each data folder must contain one
sub-folder per class. Class indices are assigned by Keras in alphabetical order
of the sub-folder names.

  <data_dir>/
      <class_1>/   image files
      <class_2>/   image files
      <class_3>/   image files

  - ClientNode.py reads ONE client's data folder (edit the data_dir variable
    for each client).
  - The four baseline scripts read the POOLED data folder (edit data_dir).
  - data_dir is a hard-coded local Windows path in every script and must be
    changed to match your machine.

Pre-processing: the scripts only rescale pixel values (1/255) and resize the
images to 180 x 180. The random-rotation augmentation and histogram
equalization described in the manuscript are NOT performed by these scripts;
the images are expected to be pre-processed before they are placed in the data
folders.


--------------------------------------------------------------------------------
5. WORKFLOW
--------------------------------------------------------------------------------
5.1  Train the three clients (ClientNode.py)
     Run the script once per client. For each run, edit:
       (i)  data_dir        -> the folder of that client's images
       (ii) the output file -> Client1.pkl (USA), Client2.pkl (China) or
                               Client3.pkl (India)
            (last code cell: pickle.dump(model.get_weights(), open(...)))
     Each run trains the model, prints test-set metrics and saves the trained
     weights to the chosen .pkl file in the working directory.

5.2  Aggregate at the server (ServerNode.py)
     Copy Client1.pkl, Client2.pkl and Client3.pkl into the folder from which
     ServerNode.py is run, then run it. The script loads the three weight sets
     and writes the element-wise average:

         W_central = (1/3) * (W_client1 + W_client2 + W_client3)

     to Central_Server.pkl. Information exchange between clients and server is
     file-based (pickled weights); no raw images are used by the server.

5.3  Train the baselines (Full_Data_*.py)
     Edit data_dir to point to the pooled data folder and run the script of the
     required architecture. Each script trains the model, prints train/test
     scores and prints a per-class classification report on the test split.

5.4  Loading the aggregated weights (example, not part of the scripts)
     Rebuild the ClientNode.py architecture (same layers, including the CBAM
     function), then:

         import pickle
         central = pickle.load(open('Central_Server.pkl', 'rb'))
         model.set_weights(list(central))


--------------------------------------------------------------------------------
6. MODEL AND TRAINING CONFIGURATION
--------------------------------------------------------------------------------
6.1  Proposed classifier (ClientNode.py)

  Backbone : four blocks of Conv2D (16, 32, 64, 128 filters; 3 x 3; ReLU)
             each followed by 2 x 2 max pooling
             (feature map after block 4: 9 x 9 x 128)
  CBAM     : applied after the fourth block, reduction ratio 8
             - channel attention: global average and global max pooling ->
               shared two-layer MLP (128 -> 16 -> 128, no bias) -> sum ->
               sigmoid -> channel-wise rescaling
             - spatial attention: channel-wise mean and max maps ->
               concatenation -> 7 x 7 convolution (1 filter, 'same' padding,
               no bias) -> sigmoid -> spatial rescaling
  Head     : Flatten -> Dense 550 (ReLU) -> Dropout 0.1
                     -> Dense 400 (ReLU) -> Dropout 0.3
                     -> Dense 200 (ReLU) -> Dropout 0.2
                     -> Dense 3 (softmax)
  Optimizer: Adam (Keras default settings)
  Loss     : SparseCategoricalCrossentropy(from_logits=True), integer labels
  Training : 40 epochs, batch size 1000, shuffling on
  Split    : train_test_split, test_size = 0.3, random_state = 30
  Seeds    : numpy seed 1, tf.random.set_seed(2),
             tf.compat.v1.set_random_seed(2019); dropout seed 2019
  Metrics  : accuracy, macro-averaged precision, recall (sensitivity) and F1
             on the test split

6.2  Baselines (all use Adam with default settings, categorical cross-entropy
     with one-hot labels, and a 70/30 train/test split)

  Script                      Initialisation / trainable layers      Epochs  Batch  Split seed
  --------------------------  -------------------------------------  ------  -----  ----------
  Full_Data_DenseNet.py       Built from scratch (3 dense blocks,       10     100       40
                              4 layers/block, growth rate 12,
                              dropout 0.2); all layers trainable
  Full_Data_InceptionV3.py    weights=None (random initialisation),     10     100       35
                              base layers frozen, truncated at
                              'mixed7'; GlobalMaxPooling -> Dense 1024
                              -> BatchNorm -> Dropout 0.5 -> Dense 3
  Full_Data_ResNet50.py       ImageNet weights; all layers trainable;   10     100       30
                              GlobalAveragePooling -> Dense 512
                              -> Dropout 0.5 -> Dense 3
  Full_Data_VGG16.py          ImageNet (Keras default) weights; base     5     100       25
                              frozen; Flatten -> Dropout 0.5 -> Dense 3

  Metrics: model.evaluate (train/test) and sklearn classification_report
           (per-class and averaged precision, recall, F1) on the test split.


--------------------------------------------------------------------------------
7. OUTPUTS
--------------------------------------------------------------------------------
  Client1.pkl, Client2.pkl, Client3.pkl   Trained client weights (ClientNode.py)
  Central_Server.pkl                      Averaged weights (ServerNode.py)
  Console output                          Start/end timestamps, model summary,
                                          train/test scores and test metrics


--------------------------------------------------------------------------------
8. NOTES AND SCOPE
--------------------------------------------------------------------------------
  1. Images are loaded with flow_from_directory(batch_size=4000) followed by a
     single .next() call in every script, so each run uses one shuffled batch
     of at most 4000 images, which is then split 70/30.
  2. Each script uses a single hold-out split (train_test_split without
     stratification; the random_state differs between scripts). The scripts do
     not perform k-fold cross-validation or repeated runs, and do not compute
     standard deviations.
  3. ClientNode.py uses class_mode='binary' together with a 3-unit softmax
     output and a sparse loss; the labels are handled as integer class indices
     (0, 1, 2).
  4. ClientNode.py applies a softmax in the output layer while the loss is
     created with from_logits=True.
  5. The comment "Squeeze and Excitation Attention Layer" next to the CBAM(x)
     call in ClientNode.py is outdated; the code implements CBAM.
  6. ServerNode.py performs one aggregation step with equal weights (1/3 per
     client). It does not include code for sending the aggregated weights back
     to the clients, for further communication rounds, or for evaluating the
     central model on out-of-country data. The 'socket' module is imported but
     not used.
  7. ServerNode.py applies np.array() to lists of weight tensors of different
     shapes. NumPy >= 1.24 raises a ValueError for such ragged input unless
     dtype=object is specified; use an older NumPy release or add
     dtype=object.
  8. Full_Data_DenseNet.py defines `classes = 2`, which is not used; the output
     layer has 3 units.


--------------------------------------------------------------------------------
9. CITATION AND LICENSE
--------------------------------------------------------------------------------
  Citation : [to be added after publication]
  License  : [to be added]
  Contact  : [to be added]
================================================================================
