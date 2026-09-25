import numpy as np
import mthree
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler


# Experimental configuration

# State parameters: (alpha, theta, phi), specified in degrees
test_states = [
    (np.deg2rad(15), np.deg2rad(25),  np.deg2rad(30)),
    (np.deg2rad(25), np.deg2rad(65),  np.deg2rad(45)),
    (np.deg2rad(30), np.deg2rad(120), np.deg2rad(60)),
    (np.deg2rad(40), np.deg2rad(35),  np.deg2rad(75)),
    (np.deg2rad(45), np.deg2rad(85),  np.deg2rad(30)),
    (np.deg2rad(50), np.deg2rad(140), np.deg2rad(45)),
    (np.deg2rad(60), np.deg2rad(20),  np.deg2rad(60)),
    (np.deg2rad(65), np.deg2rad(65),  np.deg2rad(75)),
    (np.deg2rad(70), np.deg2rad(110), np.deg2rad(30)),
    (np.deg2rad(78), np.deg2rad(35),  np.deg2rad(45)),
    (np.deg2rad(82), np.deg2rad(85),  np.deg2rad(60)),
    (np.deg2rad(85), np.deg2rad(145), np.deg2rad(75)),]

shots = 1_000
n_reps = 20
n_states = len(test_states)
n_coords = 5
circuits_per_rep = n_states * n_coords

backend = service.backend("ibm_kingston")

print(f"Backend: {backend.name}")


# Fixed physical layouts:
# logical q0 -> physical 49
# logical q1 -> physical 38
# embedding q2 -> physical 50

pm_2q = generate_preset_pass_manager(backend=backend,optimization_level=3,initial_layout=[49, 38],)

pm_3q = generate_preset_pass_manager(backend=backend,optimization_level=3,initial_layout=[49, 38, 50],)


# Circuit construction
#Prepare the original two-qubit state.
def prepare_state(alpha, theta, phi):
    
    qc = QuantumCircuit(2)
    qc.ry(alpha, 0)
    qc.cry(theta, 0, 1)
    qc.p(phi, 0)

    return qc

#Measurement circuit for x0, x1, or x2.
def make_original_measurement(alpha, theta, phi, basis):
    qc = QuantumCircuit(2, 1)
    qc.ry(alpha, 0)
    qc.cry(theta, 0, 1)
    qc.p(phi, 0)

    if basis == "X":
        qc.h(0)

    elif basis == "Y":
        qc.sdg(0)
        qc.h(0)

    elif basis != "Z":
        raise ValueError("basis must be 'X', 'Y', or 'Z'")

    qc.measure(0, 0)

    return qc

#Real-embedding measurement circuit for x3 or x4.
def make_embedded_measurement(alpha, theta, phi, pauli):
    qc = QuantumCircuit(3, 3)
    qc.ry(alpha, 0)
    qc.cry(theta, 0, 1)
    qc.cry(2 * phi, 0, 2)

    # Qiskit Pauli strings are ordered as q2 q1 q0.
    for qubit, operator in enumerate(reversed(pauli)):

        if operator == "X":
            qc.h(qubit)

        elif operator == "Y":
            qc.sdg(qubit)
            qc.h(qubit)

        elif operator not in ("I", "Z"):
            raise ValueError(f"Unsupported Pauli operator: {operator}")

    qc.measure([0, 1, 2], [0, 1, 2])

    return qc


# Build and transpile the experiment

base_isa_circuits = []

for alpha, theta, phi in test_states:

    circuits_2q = [make_original_measurement(alpha, theta, phi, "Z"),make_original_measurement(alpha, theta, phi, "X"),make_original_measurement(alpha, theta, phi, "Y"),]

    circuits_3q = [make_embedded_measurement(alpha, theta, phi, "ZYY"),make_embedded_measurement(alpha, theta, phi, "XYY"),]

    base_isa_circuits.extend(pm_2q.run(circuit)for circuit in circuits_2q)

    base_isa_circuits.extend(pm_3q.run(circuit) for circuit in circuits_3q)


# Reuse the same transpiled 60-circuit block in all 20 repetitions.
all_isa_circuits = base_isa_circuits * n_reps

print(f"Number of states:        {n_states}")
print(f"Circuits per state:      {n_coords}")
print(f"Circuits per repetition: {len(base_isa_circuits)}")
print(f"Number of repetitions:   {n_reps}")
print(f"Total circuit instances: {len(all_isa_circuits)}")
print(f"Shots per circuit:       {shots}")


# Hardware execution and M3 readout mitigation

mappings = [mthree.utils.final_measurement_mapping(circuit) for circuit in all_isa_circuits]

measured_qubits = sorted({qubit for mapping in mappings for qubit in mapping.keys()})

print(f"Physical qubits requiring calibration: {measured_qubits}")


sampler = Sampler(mode=backend)

job = sampler.run(all_isa_circuits,shots=shots,)

print(f"Job submitted: {job.job_id()}")


mit = mthree.M3Mitigation(backend)

mit.cals_from_system(measured_qubits,rep_delay=None,)

print("M3 calibration complete.")


result = job.result()

raw_counts = [ pub_result.data.c.get_counts() for pub_result in result]

print(f"Retrieved {len(raw_counts)} raw circuit results.")


mitigated_counts = [mit.apply_correction(counts, mapping)for counts, mapping in zip(raw_counts, mappings)]

print("M3 mitigation complete.")


# Expectation values
#Single-qubit Pauli expectation value.
def expectation_1q(distribution):
    return (distribution.get("0", 0) - distribution.get("1", 0))

#Three-qubit Pauli expectation value from bitstring parity.
def expectation_3q(distribution):
    value = 0.0

    for bitstring, probability in distribution.items():
        bits = bitstring.replace(" ", "")
        sign = 1 if bits.count("1") % 2 == 0 else -1
        value += sign * probability

    return value

#Convert raw counts to normalized probabilities.
def counts_to_probabilities(counts):
    total = sum(counts.values())

    return {bitstring: count / total for bitstring, count in counts.items()}


def raw_exp_1q(counts):
    return expectation_1q(counts_to_probabilities(counts))


def raw_exp_3q(counts):
    return expectation_3q(counts_to_probabilities(counts))


#Theoretical Hopf coordinates

theory_results = []

for alpha, theta, phi in test_states:

    psi = Statevector.from_instruction(prepare_state(alpha, theta, phi))

    x0 = np.real(psi.expectation_value(SparsePauliOp("IZ")))

    x1 = np.real(psi.expectation_value(SparsePauliOp("IX")))

    x2 = np.real(psi.expectation_value(SparsePauliOp("IY")))

    # Convert Qiskit's |q1 q0> ordering to the convention
    # used for the Hopf-coordinate amplitudes.
    a0, a1, a2, a3 = psi.data[[0, 2, 1, 3]]

    pi2 = a1 * a2 - a0 * a3

    x3 = 2 * np.real(pi2)
    x4 = 2 * np.imag(pi2)

    theory_results.append([x0, x1, x2, x3, x4])


theory_results = np.asarray(theory_results)


# Experimental Hopf coordinates

raw_results = np.zeros((n_reps, n_states, n_coords))

mitigated_results = np.zeros((n_reps, n_states, n_coords))


for rep in range(n_reps):

    for state in range(n_states):

        k = (rep * circuits_per_rep+ state * n_coords)

        raw_results[rep, state] = [raw_exp_1q(raw_counts[k]),raw_exp_1q(raw_counts[k + 1]),raw_exp_1q(raw_counts[k + 2]),raw_exp_3q(raw_counts[k + 3]),raw_exp_3q(raw_counts[k + 4]),]

        mitigated_results[rep, state] = [expectation_1q(mitigated_counts[k]),expectation_1q(mitigated_counts[k + 1]),expectation_1q(mitigated_counts[k + 2]),expectation_3q(mitigated_counts[k + 3]),expectation_3q(mitigated_counts[k + 4]),]


print(f"Raw results shape:       {raw_results.shape}")
print(f"M3 results shape:        {mitigated_results.shape}")
print(f"Theory results shape:    {theory_results.shape}")


# Statistics across the 20 repetitions

raw_mean = np.mean(raw_results, axis=0)
mit_mean = np.mean(mitigated_results, axis=0)

raw_sd = np.std(raw_results, axis=0, ddof=1)
mit_sd = np.std(mitigated_results, axis=0, ddof=1)


# Hopf identity: x0² + x1² + x2² + x3² + x4²
raw_S = np.sum(raw_results**2, axis=2)
mit_S = np.sum(mitigated_results**2, axis=2)

raw_S_mean = np.mean(raw_S, axis=0)
mit_S_mean = np.mean(mit_S, axis=0)

raw_S_sd = np.std(raw_S, axis=0, ddof=1)
mit_S_sd = np.std(mit_S, axis=0, ddof=1)


# D, V, and C for every hardware repetition
raw_D = np.abs(raw_results[:, :, 0])
raw_V = np.sqrt(raw_results[:, :, 1]**2+ raw_results[:, :, 2]**2)
raw_C = np.sqrt(raw_results[:, :, 3]**2+ raw_results[:, :, 4]**2)

mit_D = np.abs(mitigated_results[:, :, 0])
mit_V = np.sqrt(mitigated_results[:, :, 1]**2+ mitigated_results[:, :, 2]**2)
mit_C = np.sqrt(mitigated_results[:, :, 3]**2+ mitigated_results[:, :, 4]**2)


raw_DVC_mean = np.column_stack([np.mean(raw_D, axis=0),np.mean(raw_V, axis=0),np.mean(raw_C, axis=0),])

mit_DVC_mean = np.column_stack([np.mean(mit_D, axis=0),np.mean(mit_V, axis=0),np.mean(mit_C, axis=0),])

raw_DVC_sd = np.column_stack([np.std(raw_D, axis=0, ddof=1),np.std(raw_V, axis=0, ddof=1),np.std(raw_C, axis=0, ddof=1),])

mit_DVC_sd = np.column_stack([np.std(mit_D, axis=0, ddof=1),np.std(mit_V, axis=0, ddof=1),np.std(mit_C, axis=0, ddof=1),])


# ============================================================
# Summary
# ============================================================

print("\n20-repetition IBM experiment")
print("=" * 50)

for state in range(n_states):

    print(f"\nState S{state + 1:02d}")

    print("Theory:   ",np.round(theory_results[state], 4))

    print("Raw:      ",np.round(raw_mean[state], 4),"±",np.round(raw_sd[state], 4))

    print("M3:       ",np.round(mit_mean[state], 4),"±",np.round(mit_sd[state], 4))

    print(f"Raw Σxi²: {raw_S_mean[state]:.4f} "f"± {raw_S_sd[state]:.4f}")

    print(f"M3  Σxi²: {mit_S_mean[state]:.4f} "f"± {mit_S_sd[state]:.4f}")


raw_mae = np.mean(np.abs(raw_mean - theory_results),axis=0,)

mit_mae = np.mean(np.abs(mit_mean - theory_results),axis=0,)


print("\nGlobal summary")
print("=" * 50)

print("Raw MAE [x0, x1, x2, x3, x4]:",np.round(raw_mae, 5))

print("M3 MAE  [x0, x1, x2, x3, x4]:",np.round(mit_mae, 5))

print(f"Mean raw Σxi²: "f"{np.mean(raw_S_mean):.5f}")

print(f"Mean M3 Σxi²:  "f"{np.mean(mit_S_mean):.5f}")

print(f"Mean raw |Σxi² - 1|: "f"{np.mean(np.abs(raw_S_mean - 1)):.5f}")

print(f"Mean M3 |Σxi² - 1|:  " f"{np.mean(np.abs(mit_S_mean - 1)):.5f}")
