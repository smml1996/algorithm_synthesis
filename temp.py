
def algorithm4(qc: QuantumCircuit, basis_gates, cbits: ClassicalRegister):
	'''fake_tokyo-8'''
	CX(0,2);CX(1,2)
	if MEASURE(2) == 0:
		if MEASURE(2) == 0:
			if MEASURE(2) == 0:
				if MEASURE(2) == 1:
					if MEASURE(2) == 1:
						X(0)
			else:
				if MEASURE(2) == 0:
					if MEASURE(2) == 1:
						X(0)
				else:
					X(0)
		else:
			if MEASURE(2) == 0:
				if MEASURE(2) == 0:
					if MEASURE(2) ==1:
						X(0)
				else:
					X(0)
			else:
				X(0)
	else:
		X(0)
		CX(0,2);CX(1,2)
		if MEASURE(2) == 0:
			if MEASURE(2) == 0:
				X(0)


