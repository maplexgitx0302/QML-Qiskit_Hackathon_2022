import os
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN

################################################################

# target_dist = np.random.lognormal
# dist_para = {
#     "mean"  : 1,
#     "sigma" : 1,
# }

target_dist = np.random.normal
dist_para = {
    "loc"   : 3.5,
    "scale" : 1,
}

# target_dist = np.random.poisson
# dist_para = {
#     "lam"   : 4,
# }

################################################################

# Number training data samples
N = 1000
# Load data samples from log-normal distribution with mean=1 and standard deviation=1
real_data = target_dist(size=N, **dist_para)

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0.0, 7.0])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [3]
k = len(num_qubits)

def create_qgan(num_epochs, seed, reps, hadamard=True):
    # Set number of training epochs
    # Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
    # Batch size
    batch_size = 100

    # Initialize qGAN
    qgan = QGAN(real_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)

    # Set quantum instance to run the quantum generator
    run_real_device = False
    if run_real_device:
        ibmq_backend = "ibmq_manila"
        TOKEN = "a694317815288639dcb5e83804a28fa64cc2d1d55a418f163313be9ea885bfd27268906d593af84768f21c04a036491053cb627a70d786c3849ab4907753fb6e"
        IBMQ.save_account(TOKEN, overwrite=True) 
        provider = IBMQ.load_account()
        backend = provider.get_backend(ibmq_backend)
    else:
        backend = BasicAer.get_backend("statevector_simulator")

    algorithm_globals.random_seed = seed
    np.random.seed = seed
    qgan.seed = seed
    quantum_instance = QuantumInstance(
        backend=backend, seed_transpiler=seed, seed_simulator=seed
    )

    # Set entangler map
    entangler_map = [(0,1), (1,2)]

    # Set an initial state for the generator circuit as a uniform distribution
    # This corresponds to applying Hadamard gates on all qubits
    init_dist = QuantumCircuit(sum(num_qubits))
    if hadamard:
        init_dist.h(init_dist.qubits)

    # Set the ansatz circuit
    ansatz = TwoLocal(int(np.sum(num_qubits)), ["ry", "rz", "ry"], "cx", entanglement=entangler_map, reps=reps)

    # Set generator's initial parameters - in order to reduce the training time and hence the
    # total running time for this notebook
    # init_params = [3.0, 1.0, 1.0, 0.6, 1.6, 2.6]
    init_params = np.random.rand((reps+1)*3*3) * np.pi * 2

    # You can increase the number of training epochs and use random initial parameters.
    # init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi

    # Set generator circuit by adding the initial distribution infront of the ansatz
    g_circuit = ansatz.compose(init_dist, front=True)
    para_dict = {}
    for i in range(len(g_circuit.parameters)):
        para_dict[g_circuit.parameters[i]] = init_params[i]
    g_circuit = g_circuit.bind_parameters(para_dict)

    # Set quantum generator
    # qgan.set_generator(generator_circuit=g_circuit)
    qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
    # The parameters have an order issue that following is a temp. workaround
    qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
    # Set classical discriminator neural network
    discriminator = NumPyDiscriminator(len(num_qubits))
    qgan.set_discriminator(discriminator)
    return qgan, quantum_instance

# Run qGAN
dir_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{target_dist.__name__}")
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
num_rounds = 50
reps_list  = [2]
h_list     = [False]
epoch_list = [5]

entropy_record = {}
for reps, hadamard, num_epochs in it.product(reps_list, h_list, epoch_list):
    fig, ax = plt.subplots(num_rounds, 4, figsize=(6*4, 5*num_rounds))
    for round in range(num_rounds):
        file_prefix = os.path.join(dir_path, f"result_U_H{hadamard}_{target_dist.__name__}_r{num_rounds}_e{num_epochs}_reps{reps}")
        print(f"\nReps {reps} | Hadamard {hadamard} | num_epochs {num_epochs}:")
        qgan, quantum_instance = create_qgan(num_epochs, round, reps, hadamard)
        result = qgan.run(quantum_instance)
        for key, value in result.items():
            print(f"  {key} : {value}")
            if key == "params_g":
                np.save(f"{file_prefix}_round{round}.npy", value)
            if key == "rel_entr":
                entropy_record[f"{file_prefix}+_round{round}"] = value

        if num_rounds == 1:
            axs = ax
        else:
            axs = ax[round]

        # Plot progress w.r.t the generator's and the discriminator's loss function
        t_steps = np.arange(num_epochs)
        axs[0].set_title("Progress in the loss function")
        axs[0].plot(
            t_steps, qgan.g_loss, label="Generator loss function", color="mediumvioletred", linewidth=2
        )
        axs[0].plot(
            t_steps, qgan.d_loss, label="Discriminator loss function", color="rebeccapurple", linewidth=2
        )
        axs[0].grid()
        axs[0].legend(loc="best")
        axs[0].set_xlabel("time steps")
        axs[0].set_ylabel("loss")

        # Plot progress w.r.t relative entropy
        axs[1].set_title(f"Relative Entropy Round {round}")
        axs[1].plot(
            np.linspace(0, num_epochs, len(qgan.rel_entr)), qgan.rel_entr, color="mediumblue", lw=4, ls=":"
        )
        axs[1].grid()
        axs[1].set_xlabel("time steps")
        axs[1].set_ylabel("relative entropy")

        # Plot the CDF of the resulting distribution against the target distribution, i.e. log-normal
        target_d = target_dist(size=100000, **dist_para)
        target_d = np.round(target_d)
        target_d = target_d[target_d <= bounds[1]]
        temp = []
        for i in range(int(bounds[1] + 1)):
            temp += [np.sum(target_d == i)]
        target_d = np.array(temp / sum(temp))

        axs[2].set_title("CDF (Cumulative Distribution Function)")
        samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
        samples_g = np.array(samples_g)
        samples_g = samples_g.flatten()
        num_bins = len(prob_g)
        axs[2].bar(samples_g, np.cumsum(prob_g), color="royalblue", width=0.8, label="simulation")
        axs[2].plot(
            np.cumsum(target_d), "-o", label=target_dist.__name__, color="deepskyblue", linewidth=4, markersize=12
        )
        axs[2].set_xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
        axs[2].grid()
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("p(x)")
        axs[2].legend(loc="best")
        
        # Plot the PDF of the resulting distribution against the target distribution, i.e. log-normal
        target_d = target_dist(size=100000, **dist_para)
        target_d = np.round(target_d)
        target_d = target_d[target_d <= bounds[1]]
        temp = []
        for i in range(int(bounds[1] + 1)):
            temp += [np.sum(target_d == i)]
        target_d = np.array(temp / sum(temp))

        axs[3].set_title("PDF (Probability Distribution Function)")
        samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
        samples_g = np.array(samples_g)
        samples_g = samples_g.flatten()
        num_bins = len(prob_g)
        axs[3].bar(samples_g, prob_g, color="royalblue", width=0.8, label="simulation")
        axs[3].plot(
            target_d, "-o", label=target_dist.__name__, color="deepskyblue", linewidth=4, markersize=12
        )
        axs[3].set_xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
        axs[3].grid()
        axs[3].set_xlabel("x")
        axs[3].set_ylabel("p(x)")
        axs[3].legend(loc="best")


    fig_path = os.path.join(dir_path, f"{file_prefix}.jpg")
    plt.savefig(fig_path)

np.save(os.path.join(dir_path, f"{file_prefix}_entropy.npy"), entropy_record, allow_pickle=True)