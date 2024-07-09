from inferential_gates import * 
from gates_qlm import *

from qat.qpus import CLinalg


def test_AND(output_precision = 0.6):
    qpu = CLinalg()
    object_and = AND(qpu, output_precision)
    object_and.fit()

    # Components |1> o |a> and |b>
    alpha_a = object_and.alpha_a
    alpha_b = object_and.alpha_b

    # Precendent states of AND gate
    state_a = PRECISION(qpu, alpha_a)
    state_a.fit()
    state_b = PRECISION(qpu, alpha_b)
    state_b.fit()

    
    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(precision_gate(state_a.alpha_a), register[0])
    routine.apply(precision_gate(state_b.alpha_a), register[1])
    routine.apply(and_gate(), register)
    display(routine)
    job = routine.to_circ().to_job(
        nbshots=0,
        qubits=[2]
    )


    pdf = solve_qjob(job, qpu)
    measured_precision = float(pdf[pdf["Int"] == 1]["Probability"])

    print("Desired Precision: {}. Gotten precision: {}".format(
        output_precision, measured_precision)
    )
    absolute_error = abs(output_precision - measured_precision)
    return output_precision, measured_precision, absolute_error


def test_OR(output_precision = 0.6):
    qpu = CLinalg()
    object_and = OR(qpu, output_precision)
    object_and.fit()

    # Components |1> o |a> and |b>
    alpha_a = object_and.alpha_a
    alpha_b = object_and.alpha_b

    # Precendent states of OR gate
    state_a = PRECISION(qpu, alpha_a)
    state_a.fit()
    state_b = PRECISION(qpu, alpha_b)
    state_b.fit()

    
    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(precision_gate(state_a.alpha_a), register[0])
    routine.apply(precision_gate(state_b.alpha_a), register[1])
    routine.apply(or_gate(), register)
    display(routine, max_depth=None)
    job = routine.to_circ().to_job(
        nbshots=0,
        qubits=[2]
    )


    pdf = solve_qjob(job, qpu)
    measured_precision = float(pdf[pdf["Int"] == 1]["Probability"])

    print("Desired Precision: {}. Gotten precision: {}".format(
        output_precision, measured_precision)
    )
    absolute_error = abs(output_precision - measured_precision)
    return output_precision, measured_precision, absolute_error

def test_NOT(output_precision = 0.6):
    qpu = CLinalg()
    object_and = NOT(qpu, output_precision)
    object_and.fit()

    # Components |1> of |a>
    alpha_a = object_and.alpha_a

    # Precendent states of NOT gate
    state_a = PRECISION(qpu, alpha_a)
    state_a.fit()

    
    routine = qlm.QRoutine()
    register = routine.new_wires(2)
    routine.apply(precision_gate(state_a.alpha_a), register[0])
    routine.apply(not_gate(), register)
    display(routine, max_depth=None)
    job = routine.to_circ().to_job(
        nbshots=0,
        qubits=[1]
    )


    pdf = solve_qjob(job, qpu)
    measured_precision = float(pdf[pdf["Int"] == 1]["Probability"])

    print("Desired Precision: {}. Gotten precision: {}".format(
        output_precision, measured_precision)
    )
    absolute_error = abs(output_precision - measured_precision)
    return output_precision, measured_precision, absolute_error


def test_RULE(output_precision = 0.6):
    qpu = CLinalg()
    object_and = RULE(qpu, output_precision)
    object_and.fit()

    # Components |1> o |a> and |b>
    alpha_a = object_and.alpha_a
    alpha_b = object_and.alpha_b

    # Precendent states of OR gate
    state_a = PRECISION(qpu, alpha_a)
    state_a.fit()

    
    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(precision_gate(state_a.alpha_a), register[0])
    routine.apply(rule_gate(alpha_b), register)
    display(routine, max_depth=None)
    job = routine.to_circ().to_job(
        nbshots=0,
        qubits=[2]
    )


    pdf = solve_qjob(job, qpu)
    measured_precision = float(pdf[pdf["Int"] == 1]["Probability"])

    print("Desired Precision: {}. Gotten precision: {}".format(
        output_precision, measured_precision)
    )
    absolute_error = abs(output_precision - measured_precision)
    return output_precision, measured_precision, absolute_error

def test_circuit(output_precision = 0.6):
    qpu = CLinalg()

    lista_final = []


    #  Final Rule
    object_rule = RULE(qpu, output_precision)
    object_rule.fit()
    lista_final.append(["Rule", None, None, object_rule.alpha_b])


    # Not Gate
    object_not = NOT(qpu, object_rule.alpha_a)
    object_not.fit()
    # not_state = PRECISION(qpu, object_not.alpha_a)
    # not_state.fit()
    lista_final.append(["Not", None, None, None])

    # And Gate
    object_and = AND(qpu, object_not.alpha_a)
    object_and.fit()
    lista_final.append(["And", None, None, None])
    # Second precedent of And Gate
    # and_state_a = PRECISION(qpu, object_and.alpha_a)
    # and_state_a.fit()
    and_state_b = PRECISION(qpu, object_and.alpha_b)
    and_state_b.fit()
    lista_final.append(["Precision", and_state_b.alpha_a, None, None])

    # Or Gate
    object_or = OR(qpu, object_and.alpha_a)
    object_or.fit()
    lista_final.append(["Or", None, None, None])
    state_or_a = PRECISION(qpu, object_or.alpha_a)
    state_or_a.fit()
    lista_final.append(["Precision", state_or_a.alpha_a, None, None])
    state_or_b = PRECISION(qpu, object_or.alpha_b)
    state_or_b.fit()
    lista_final.append(["Precision", state_or_b.alpha_a, None, None])

    # routine = qlm.QRoutine()
    # register = routine.new_wires(8)
    # # Initial Precisions
    # routine.apply(precision_gate(state_or_a.alpha_a), register[0])
    # routine.apply(precision_gate(state_or_b.alpha_a), register[1])

    # # Or gte
    # routine.apply(or_gate(), register[0], register[1], register[2])

    # # New state for ANd
    # routine.apply(precision_gate(and_state_b.alpha_a), register[3])

    # # And gate
    # routine.apply(and_gate(), register[2], register[3], register[4])

    # # Not gate
    # routine.apply(not_gate(), register[4], register[5])


    # # Rule gate
    # routine.apply(
    #     rule_gate(object_rule.alpha_b),
    #     register[5], register[6], register[7]
    # )

    lista_final.reverse()

    routine = qlm.QRoutine()
    for step in lista_final:
        routine = create_circuit(step, routine)

    display(routine, max_depth=None)
    job = routine.to_circ().to_job(
        nbshots=0,
        qubits=[routine.max_wire]
    )


    pdf = solve_qjob(job, qpu)
    #print(pdf)
    measured_precision = float(pdf[pdf["Int"] == 1]["Probability"])

    print("Desired Precision: {}. Gotten precision: {}".format(
        output_precision, measured_precision)
    )
    absolute_error = abs(output_precision - measured_precision)
    return output_precision, measured_precision, absolute_error


def create_circuit(step, routine):

    if step[0] == "Precision":

        routine.new_wires(1)
        routine.apply(precision_gate(step[1]), routine.max_wire)

    if  step[0] == "And":
        routine.new_wires(1)
        routine.apply(
            and_gate(),
            routine.max_wire-2,
            routine.max_wire-1,
            routine.max_wire,
        )
    if  step[0] == "Or":
        routine.new_wires(1)
        routine.apply(
            or_gate(),
            routine.max_wire-2,
            routine.max_wire-1,
            routine.max_wire,
        )
    if  step[0] == "Not":
        routine.new_wires(1)
        routine.apply(
            not_gate(),
            routine.max_wire-1,
            routine.max_wire,
        )
    if step[0] == "Rule":
        routine.new_wires(1)
        routine.new_wires(1)
        routine.apply(
            rule_gate(step[-1]),
            routine.max_wire-2,
            routine.max_wire-1,
            routine.max_wire,
        )

    return routine



if __name__ == "__main__":
    from qat.core.console import display

    test_AND(0.8)
    test_OR(0.8)
    test_NOT(0.8)
    test_RULE(0.8)
    test_circuit(0.8)

    # routine = qlm.QRoutine()


    # lista =[
    #     ["Precision", 0.16055513055328624, None, None],
    #     ['Precision', 0.1435714110111506, None, None],
    #     ['And', None, None, None]
    # ]

    # for step in lista:

    #     routine = create_circuit(step, routine)
  
    # display(routine)
    


    # lista = []
    # for step in np.linspace(0.01, 0.99):
    #     #th, meas, error = test_AND(step)
    #     th, meas, error = test_circuit(step)
    #     print(th, meas, error)
    #     lista.append(error)
    # print(max(lista))
