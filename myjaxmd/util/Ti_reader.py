"""Read in data of Ti DFT data"""
import numpy as onp
import glob


def get_np_data_dpgen(data_dir, lammpsdata_bool):
    """Function to read in and return numpy arrays
    for the dpgen data"""
    box = onp.load(data_dir + '/box.npy', allow_pickle=True)
    coord = onp.load(data_dir + '/coord.npy', allow_pickle=True)
    energy = onp.load(data_dir + '/energy.npy', allow_pickle=True)
    force = onp.load(data_dir + '/force.npy', allow_pickle=True)
    virial = onp.load(data_dir + '/virial.npy', allow_pickle=True)

    if lammpsdata_bool:
        box = onp.reshape(box, (box.shape[0], -1))
        coord = onp.reshape(coord, (coord.shape[0], -1))
        force = onp.reshape(force, (force.shape[0], -1))
        virial = onp.reshape(virial, (virial.shape[0], -1))

        # Convert to proper units
        box = box * 0.1         # Å to nm
        coord = coord * 0.1     # Å to nm
        energy = energy * 96.4853    # [eV] to [kJ/mol]
        force = force * 964.853     # [eV/Å] to [kJ/mol*nm]

    return box, coord, energy, force, virial


def get_np_data_dpgen_without_virial(data_dir):
    """Function to read in and return numpy arrays
    for the dpgen data. Without virial, as this function
    is called for all data files and gamma_line_pyr1_narrow
    does not contain virial."""
    box = onp.load(data_dir + '/box.npy')
    coord = onp.load(data_dir + '/coord.npy')
    energy = onp.load(data_dir + '/energy.npy')
    force = onp.load(data_dir + '/force.npy')

    return box, coord, energy, force

def list_of_all_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")

    surf_dpgen_dir = glob.glob(ti_dir_str + "surf_dpgen_1019_iter32/**/**/**/set.000")

    gamma_line_basal = glob.glob(ti_dir_str + "gamma_line/basal/**/**/**/**/set.000")
    gamma_line_prism = glob.glob(ti_dir_str + "gamma_line/prism1/**/**/**/**/set.000")
    # gamma_line_pyr1_narrow = glob.glob(ti_dir_str + "gamma_line/pyr1_narrow/**/set.000")   #DOES NOT CONTAIN VIRIAL!
    gamma_line_pyr2 = glob.glob(ti_dir_str + "gamma_line/pyr2/**/**/**/**/set.000")

    all_dirs = [init_bcc, init_fcc, init_hcp, init_0823_bcc, init_0823_fcc, init_0823_hcp] + \
               bulk_dpgen_1_dir + bulk_dpgen_2_dir + surf_dpgen_dir + gamma_line_basal + \
               gamma_line_prism + gamma_line_pyr2  # + gamma_line_pyr1_narrow

    return all_dirs

def list_of_HCP_init_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    all_dirs = [init_hcp, init_0823_hcp]

    return all_dirs

def list_of_BCC_init_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"


    all_dirs = [init_bcc, init_0823_bcc]

    return all_dirs

def list_of_FCC_init_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"

    all_dirs = [init_fcc, init_0823_fcc]

    return all_dirs

def list_of_all_init_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    all_dirs = [init_bcc, init_fcc, init_hcp, init_0823_bcc, init_0823_fcc, init_0823_hcp]

    return all_dirs


def list_of_all_init_data_only_portion():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    # init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    # init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    # init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    all_dirs = [init_bcc, init_fcc, init_hcp] #, init_0823_bcc, init_0823_fcc, init_0823_hcp]

    return all_dirs

def list_of_all_Bulk():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"


    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")

    all_dirs = bulk_dpgen_1_dir + bulk_dpgen_2_dir

    return all_dirs


def list_of_init_and_bulk_data():
    """Function returns a list with all directories of
    the Ti DFT data."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"
    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")

    all_dirs = [init_hcp, init_bcc, init_fcc, init_0823_hcp, init_0823_bcc, init_0823_fcc] + bulk_dpgen_1_dir + bulk_dpgen_2_dir

    return all_dirs

def list_of_Init1_and_Bulk1():
    """Function returns a list with Bulk Ti DFT data vom DP paper"""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"
    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    all_dirs = [init_hcp, init_bcc, init_fcc] + bulk_dpgen_1_dir

    return all_dirs


def list_of_bulk1_data():
    """Function returns a list with Bulk Ti DFT data vom DP paper"""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"
    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    return bulk_dpgen_1_dir


def list_of_bulk2_data():
    """Function returns a list with Bulk Ti DFT data vom DP paper"""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")
    return bulk_dpgen_2_dir


def list_of_entire_bulk_data():
    """Function returns a list with Bulk Ti DFT data vom DP paper"""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"
    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")
    return bulk_dpgen_1_dir + bulk_dpgen_2_dir


def list_of_all_data_without_virial():
    """Function returns a list with all directories of
    the Ti DFT data.
    Also contains gamma_line_pyr1_narrow for which no virial is available."""
    ti_dir_str = "data/Dataset_Ti_paper/data_0818/Ti_0722/"

    init_bcc = ti_dir_str + "init/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_fcc = ti_dir_str + "init/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_hcp = ti_dir_str + "init/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_bcc = ti_dir_str + "init_0823/bcc/POSCAR.bcc.02x02x02/02.md/sys-0016/deepmd/set.000"
    init_0823_fcc = ti_dir_str + "init_0823/fcc/POSCAR.fcc.02x02x02/02.md/sys-0032/deepmd/set.000"
    init_0823_hcp = ti_dir_str + "init_0823/hcp/POSCAR.hcp.02x02x02/02.md/sys-0016/deepmd/set.000"

    bulk_dpgen_1_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915/**/**/**/set.000")
    bulk_dpgen_2_dir = glob.glob(ti_dir_str + "bulk_dpgen_0915-2/**/**/**/set.000")

    surf_dpgen_dir = glob.glob(ti_dir_str + "surf_dpgen_1019_iter32/**/**/**/set.000")

    gamma_line_basal = glob.glob(ti_dir_str + "gamma_line/basal/**/**/**/**/set.000")
    gamma_line_prism = glob.glob(ti_dir_str + "gamma_line/prism1/**/**/**/**/set.000")
    gamma_line_pyr1_narrow = glob.glob(ti_dir_str + "gamma_line/pyr1_narrow/**/set.000")   #DOES NOT CONTAIN VIRIAL!
    gamma_line_pyr2 = glob.glob(ti_dir_str + "gamma_line/pyr2/**/**/**/**/set.000")

    all_dirs = [init_bcc, init_fcc, init_hcp, init_0823_bcc, init_0823_fcc, init_0823_hcp] + \
               bulk_dpgen_1_dir + bulk_dpgen_2_dir + surf_dpgen_dir + gamma_line_basal + \
               gamma_line_prism + gamma_line_pyr2 + gamma_line_pyr1_narrow

    return all_dirs


def adjust_property_arrays(dir_files):
    """This function takes in a list of directories that contain
    box, coord, energy, force, virial npy files and returns for each
    of these properties a list containing all entries in a new format.
    box = np.array(n_structures, np.array(3,3))
    coords = [n_structures, np.array(atoms_strucutre,3)]
    energy = np.array(n_structures)
    force = [n_structures, np.array(atoms_strucutre,3)]
    virial = np.array(n_structures, np.array(3,3))"""

    all_boxes = []
    all_coords = []
    all_energy = []
    all_force = []
    all_virial = []
    for i in range(len(dir_files)):
        box, coord, energy, force, virial = get_np_data_dpgen(dir_files[i])

        all_boxes = all_boxes + [onp.array([[j[0], j[1], j[2]], [j[3], j[4], j[5]], [j[6], j[7], j[8]]]) for j in box]

        all_virial = all_virial + [onp.array([[j[0], j[1], j[2]], [j[3], j[4], j[5]], [j[6], j[7], j[8]]]) for j in
                                   virial]

        for j in coord:
            num = int(len(j))
            all_coords.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

        for j in energy:
            all_energy.append(j)

        for j in force:
            num = int(len(j))
            all_force.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

    all_energy = onp.array(all_energy)
    all_boxes = onp.array(all_boxes)
    all_virial = onp.array(all_virial)
    return all_boxes, all_coords, all_energy, all_force, all_virial


def get_train_val_test_set(dir_files, shuffle=True, shuffle_subset=True, only_consider_len_16=False,
                           only_consider_len_32=False, only_consider_len_128=False, only_consider_len_192=False,
                           lammpsdata_bool=False, set_manual_weights=False):
    """This function takes in a list of directories that contain
        box, coord, energy, force, virial npy files and returns for each
        of these properties a train(70%), validation(10%), and test(20%) set in a new format.
        box = np.array(n_structures, np.array(3,3))
        coords = [n_structures, np.array(atoms_strucutre,3)]
        energy = np.array(n_structures)
        force = [n_structures, np.array(atoms_strucutre,3)]
        virial = np.array(n_structures, np.array(3,3))"""

    train_boxes, train_coords, train_energy, train_force, train_virial = [], [], [], [], []
    val_boxes, val_coords, val_energy, val_force, val_virial = [], [], [], [], []
    test_boxes, test_coords, test_energy, test_force, test_virial = [], [], [], [], []

    if set_manual_weights:
        train_type = []
        val_type = []
        test_type = []


    total_test_len = []  # Used to know which subfolder contains how many samples (Testing on specific sets afterwards)
    # print("Be aware: Only subsamples > 20 are considered!")
    # print("Be aware: Only structure of length 16 are considered!")
    for i in range(len(dir_files)):
        box, coord, energy, force, virial = get_np_data_dpgen(dir_files[i], lammpsdata_bool)
        if set_manual_weights:
            if dir_files[i][40:44] == 'init':
                temp_type = energy.shape[0] * [0]
            elif dir_files[i][40:44] == 'bulk':
                temp_type = energy.shape[0] * [1]

        # if only_consider_len_16:
        if int(len(coord[0])/3) == 16:
            indices_short = [int(len(i)/3) == 16 for i in coord]
            box = box[indices_short]
            coord = coord[indices_short]
            energy = energy[indices_short]
            force = force[indices_short]
            virial = virial[indices_short]
        # elif only_consider_len_32:
        elif int(len(coord[0])/3) == 32:
            indices_short = [int(len(i)/3) == 32 for i in coord]
            box = box[indices_short]
            coord = coord[indices_short]
            energy = energy[indices_short]
            force = force[indices_short]
            virial = virial[indices_short]
        elif only_consider_len_128:
            indices_short = [int(len(i)/3) == 128 for i in coord]
            box = box[indices_short]
            coord = coord[indices_short]
            energy = energy[indices_short]
            force = force[indices_short]
            virial = virial[indices_short]
        elif only_consider_len_192:
            indices_short = [int(len(i) / 3) == 192 for i in coord]
            box = box[indices_short]
            coord = coord[indices_short]
            energy = energy[indices_short]
            force = force[indices_short]
            virial = virial[indices_short]
        elif int(len(coord[0])/3) == 256:
            indices_short = [int(len(i) / 3) == 256 for i in coord]
            box = box[indices_short]
            coord = coord[indices_short]
            energy = energy[indices_short]
            force = force[indices_short]
            virial = virial[indices_short]


        # Check that subset size is larger than 20, else do not consider
        if len(energy) > 1:

            temp_boxes, temp_coords, temp_energy, temp_force, temp_virial = [], [], [], [], []

            temp_boxes = [onp.array([[j[0], j[1], j[2]], [j[3], j[4], j[5]], [j[6], j[7], j[8]]]) for j in box]

            temp_virial = [onp.array([[j[0], j[1], j[2]], [j[3], j[4], j[5]], [j[6], j[7], j[8]]]) for j in virial]

            for j in coord:
                num = int(len(j))
                temp_coords.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

            for j in energy:
                temp_energy.append(j)

            for j in force:
                num = int(len(j))
                temp_force.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

            # Split data into train(70%), validation(10%), and test(20%) - train and validation are >= 1
            data_len = len(temp_boxes)
            val_len = int(data_len * 0.1) if int(data_len * 0.1) >= 1. else 1
            test_len = int(data_len * 0.2) if int(data_len * 0.2) >= 1. else 1
            total_test_len.append(test_len)
            train_len = data_len - val_len - test_len

            # Shuffle subset before assigning training, test and validation
            if shuffle_subset:
                # Turn list into onp.array to shuffle
                temp_energy = onp.array(temp_energy)
                temp_force = onp.array(temp_force)
                temp_virial = onp.array(temp_virial)
                temp_coords = onp.array(temp_coords)
                temp_boxes = onp.array(temp_boxes)

                # Shuffle
                shuffle_indices = onp.arange(len(energy))
                onp.random.shuffle(shuffle_indices)
                temp_energy = temp_energy[shuffle_indices]
                temp_force = temp_force[shuffle_indices]
                temp_virial = temp_virial[shuffle_indices]
                temp_coords = temp_coords[shuffle_indices]
                temp_boxes = temp_boxes[shuffle_indices]

                # Back to lists
                temp_energy = list(temp_energy)
                temp_force = list(temp_force)
                temp_virial = list(temp_virial)
                temp_coords = list(temp_coords)
                temp_boxes = list(temp_boxes)

            train_boxes = train_boxes + temp_boxes[:train_len]
            val_boxes = val_boxes + temp_boxes[train_len:train_len+val_len]
            test_boxes = test_boxes + temp_boxes[train_len+val_len:]

            train_coords = train_coords + temp_coords[:train_len]
            val_coords = val_coords + temp_coords[train_len:train_len + val_len]
            test_coords = test_coords + temp_coords[train_len + val_len:]

            train_energy = train_energy + temp_energy[:train_len]
            val_energy = val_energy + temp_energy[train_len:train_len + val_len]
            test_energy = test_energy + temp_energy[train_len + val_len:]

            train_force = train_force + temp_force[:train_len]
            val_force = val_force + temp_force[train_len:train_len + val_len]
            test_force = test_force + temp_force[train_len + val_len:]

            train_virial = train_virial + temp_virial[:train_len]
            val_virial = val_virial + temp_virial[train_len:train_len + val_len]
            test_virial = test_virial + temp_virial[train_len + val_len:]

            if set_manual_weights:
                train_type = train_type + temp_type[:train_len]
                val_type = val_type + temp_type[train_len:train_len + val_len]
                test_type = test_type + temp_type[train_len + val_len:]


    train_energy = onp.array(train_energy)
    train_boxes = onp.array(train_boxes)
    train_virial = onp.array(train_virial)

    test_energy = onp.array(test_energy)
    test_boxes = onp.array(test_boxes)
    test_virial = onp.array(test_virial)

    val_energy = onp.array(val_energy)
    val_boxes = onp.array(val_boxes)
    val_virial = onp.array(val_virial)

    debug_subset = False
    if debug_subset:
        # Make dataset smaller
        train_energy = onp.array([train_energy[i] for i in range(0, len(train_energy), 50)])
        train_boxes = onp.array([train_boxes[i] for i in range(0, len(train_boxes), 50)])
        train_virial = onp.array([train_virial[i] for i in range(0, len(train_virial), 50)])
        train_coords = [train_coords[i] for i in range(0, len(train_coords), 50)]
        train_force = [train_force[i] for i in range(0, len(train_force), 50)]

        test_energy = onp.array([test_energy[i] for i in range(0, len(test_energy), 50)])
        test_boxes = onp.array([test_boxes[i] for i in range(0, len(test_boxes), 50)])
        test_virial = onp.array([test_virial[i] for i in range(0, len(test_virial), 50)])
        test_coords = [test_coords[i] for i in range(0, len(test_coords), 50)]
        test_force = [test_force[i] for i in range(0, len(test_force), 50)]

        val_energy = onp.array([val_energy[i] for i in range(0, len(val_energy), 50)])
        val_boxes = onp.array([val_boxes[i] for i in range(0, len(val_boxes), 50)])
        val_virial = onp.array([val_virial[i] for i in range(0, len(val_virial), 50)])
        val_coords = [val_coords[i] for i in range(0, len(val_coords), 50)]
        val_force = [val_force[i] for i in range(0, len(val_force), 50)]

    if shuffle:

        train_indices = onp.arange(len(train_energy))
        onp.random.shuffle(train_indices)
        train_energy = train_energy[train_indices]
        train_boxes = train_boxes[train_indices]
        train_virial = train_virial[train_indices]
        train_coords = [train_coords[i] for i in train_indices]
        train_force = [train_force[i] for i in train_indices]
        # Currently don't shuffle data to know what I'm testing on
        # test_indices = onp.arange(len(test_energy))
        # onp.random.shuffle(test_indices)
        # test_energy = test_energy[test_indices]
        # test_boxes = test_boxes[test_indices]
        # test_virial = test_virial[test_indices]
        # test_coords = [test_coords[i] for i in test_indices]
        # test_force = [test_force[i] for i in test_indices]

        val_indices = onp.arange(len(val_energy))
        onp.random.shuffle(val_indices)
        val_energy = val_energy[val_indices]
        val_boxes = val_boxes[val_indices]
        val_virial = val_virial[val_indices]
        val_coords = [val_coords[i] for i in val_indices]
        val_force = [val_force[i] for i in val_indices]

        if set_manual_weights:
            train_type = [train_type[i] for i in train_indices]
            val_type = [val_type[i] for i in val_indices]

    if set_manual_weights:
        return train_energy, train_force, train_virial, train_boxes, train_coords, \
               test_energy, test_force, test_virial, test_boxes, test_coords, \
               val_energy, val_force, val_virial, val_boxes, val_coords, total_test_len, \
               train_type, val_type, test_type
    else:
        return train_energy, train_force, train_virial, train_boxes, train_coords,\
               test_energy, test_force, test_virial, test_boxes, test_coords, \
               val_energy, val_force, val_virial, val_boxes, val_coords, total_test_len


def adjust_property_arrays_without_virial(dir_files):
    """This function takes in a list of directories that contain
    box, coord, energy, force npy files and returns for each
    of these properties a list containing all entries in a new format.
    box = np.array(n_structures, np.array(3,3))
    coords = [n_structures, np.array(atoms_strucutre,3)]
    energy = np.array(n_structures)
    force = [n_structures, np.array(atoms_strucutre,3)]
    """

    all_boxes = []
    all_coords = []
    all_energy = []
    all_force = []
    for i in range(len(dir_files)):
        box, coord, energy, force = get_np_data_dpgen_without_virial(dir_files[i])

        all_boxes = all_boxes + [onp.array([[j[0], j[1], j[2]], [j[3], j[4], j[5]], [j[6], j[7], j[8]]]) for j in box]


        for j in coord:
            num = int(len(j))
            all_coords.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

        for j in energy:
            all_energy.append(j)

        for j in force:
            num = int(len(j))
            all_force.append(onp.array([[j[k], j[k+1], j[k+2]] for k in range(0, num, 3)]))

    all_energy = onp.array(all_energy)
    all_boxes = onp.array(all_boxes)

    return all_boxes, all_coords, all_energy, all_force


def save_np_array(np_array, save_str):
    """Save numpy array as npy in save_str"""
    onp.save(save_str, np_array)


if __name__ == '__main__':
    ti_data_list = list_of_all_data()

    # NOTE: The file pyr1_narrow does not contain the virial. Currently commented out when computing
    # all properties. Comment is within adjust_property_arrays()
    all_boxes, all_coords, all_energy, all_force, all_virial = adjust_property_arrays(ti_data_list)
    print('Done')