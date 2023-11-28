from pyboy import PyBoy

pyboy = PyBoy("ROMs/Pokemon Red.gb")
state_name = "ROMs/2_Pokemon Red.gb.state"
pyboy.load_state(open(state_name, "rb"))
pyboy.set_emulation_speed(3)
while not pyboy.tick():
    pass
pyboy.stop()
